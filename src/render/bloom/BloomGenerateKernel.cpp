#include "BloomGenerateKernel.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/bloom/BloomFft.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/generate_kernel.h>

using namespace glm;
using namespace wheels;

namespace
{
constexpr uvec3 sGroupSize{16, 16, 1};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/kernel_generate.comp",
        .debugName = String{alloc, "BloomGenerateKernelCS"},
        .groupSize = sGroupSize,
    };
}

} // namespace

void BloomGenerateKernel::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void BloomGenerateKernel::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void BloomGenerateKernel::drawUi()
{
    ImGui::Checkbox("Re-generate kernel", &m_reGenerate);
}

ImageHandle BloomGenerateKernel::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const vk::Extent2D &renderExtent, BloomFft &fft, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    const uint32_t dim = std::max(
        std::bit_ceil(std::max(renderExtent.width, renderExtent.height)) / 2,
        BloomFft::sMinResolution);
    WHEELS_ASSERT(dim % sGroupSize.x == 0 && "Shader doesn't do bounds checks");
    WHEELS_ASSERT(dim % sGroupSize.y == 0 && "Shader doesn't do bounds checks");

    if (!m_reGenerate && gRenderResources.images->isValidHandle(m_kernelDft))
    {
        const vk::Extent3D previousExtent =
            gRenderResources.images->resource(m_kernelDft).extent;
        const uint32_t previousDim = std::max(
            std::bit_ceil(
                std::max(previousExtent.width, previousExtent.height)) /
                2,
            BloomFft::sMinResolution);
        if (dim == previousDim)
        {
            gRenderResources.images->preserve(m_kernelDft);
            return m_kernelDft;
        }

        gRenderResources.images->release(m_kernelDft);
    }

    ImageHandle kernel;
    {
        PROFILER_CPU_SCOPE("  GenerateKernel");

        kernel = gRenderResources.images->create(
            ImageDescription{
                .format = sIlluminationFormat,
                .width = dim,
                .height = dim,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomKernel");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images->resource(kernel).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 1>{{
                    {kernel, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  GenerateKernel");

        const vk::DescriptorSet descriptorSet =
            m_computePass.storageSet(nextFrame);

        const GenerateKernelPC pcBlock{
            .invRenderResolution = 1.f /
                                   vec2{
                                       static_cast<float>(renderExtent.width),
                                       static_cast<float>(renderExtent.height),
                                   },
        };
        const uvec3 groupCount = m_computePass.groupCount(uvec3{dim, dim, 1u});
        m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
    }

    m_kernelDft = fft.record(
        scopeAlloc.child_scope(), cb, kernel, nextFrame, false, "BloomKernel");
    gRenderResources.images->preserve(m_kernelDft);

    gRenderResources.images->release(kernel);

    return m_kernelDft;
}
