#include "GenerateKernel.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "render/bloom/Fft.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace render::bloom
{

namespace
{
constexpr uvec3 sGroupSize{16, 16, 1};

ComputePass::Shader generateShaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/generate_kernel.comp",
        .debugName = String{alloc, "BloomGenerateKernelCS"},
        .groupSize = sGroupSize,
    };
}

ComputePass::Shader prepareShaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/prepare_kernel.comp",
        .debugName = String{alloc, "BloomPrepareKernelCS"},
        .groupSize = sGroupSize,
    };
}

} // namespace

void GenerateKernel::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_generatePass.init(
        scopeAlloc.child_scope(), generateShaderDefinitionCallback);
    m_preparePass.init(WHEELS_MOV(scopeAlloc), prepareShaderDefinitionCallback);

    m_initialized = true;
}

void GenerateKernel::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_reGenerate |= m_generatePass.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        generateShaderDefinitionCallback);
    m_reGenerate |= m_preparePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, prepareShaderDefinitionCallback);
}

void GenerateKernel::drawUi()
{
    ImGui::Checkbox("Re-generate kernel", &m_reGenerate);
}

float GenerateKernel::convolutionScale() const
{
    return 2.f / static_cast<float>(m_previousKernelImageDim);
}

ImageHandle GenerateKernel::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const vk::Extent2D &renderExtent, Fft &fft, ResolutionScale resolutionScale,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    const uint32_t resolutionScaleUint = bloomResolutionScale(resolutionScale);
    const uint32_t kernelImageDim = renderExtent.height / resolutionScaleUint;
    const uint32_t dim = std::max(
        std::bit_ceil(std::max(renderExtent.width, renderExtent.height)) /
            resolutionScaleUint,
        Fft::sMinResolution);
    WHEELS_ASSERT(
        dim % sGroupSize.x == 0 && "Prepare shader doesn't do bounds checks");
    WHEELS_ASSERT(
        dim % sGroupSize.y == 0 && "Prepare shader doesn't do bounds checks");

    if (gRenderResources.images->isValidHandle(m_kernelDft))
    {
        if (!m_reGenerate)
        {
            if (kernelImageDim == m_previousKernelImageDim)
            {
                gRenderResources.images->preserve(m_kernelDft);
                return m_kernelDft;
            }
        }
        gRenderResources.images->release(m_kernelDft);
    }

    const ImageHandle kernelImage =
        recordGenerate(scopeAlloc.child_scope(), cb, kernelImageDim, nextFrame);

    recordPrepare(
        scopeAlloc.child_scope(), cb, dim, fft, kernelImage, nextFrame);

    gRenderResources.images->release(kernelImage);

    return m_kernelDft;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) private
ImageHandle GenerateKernel::recordGenerate(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const uint32_t dim,
    const uint32_t nextFrame)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    ImageHandle kernel;
    {
        PROFILER_CPU_SCOPE("  GenerateKernel");

        kernel = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = Fft::sFftFormat,
                .width = dim,
                .height = dim,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomKernelImageCentered");

        const vk::DescriptorSet descriptorSet = m_generatePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images->resource(kernel).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 1>{{
                    {kernel, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  GenerateKernel");

        const uvec3 groupCount = m_generatePass.groupCount(uvec3{dim, dim, 1u});
        m_generatePass.record(cb, groupCount, Span{&descriptorSet, 1});
    }
    m_previousKernelImageDim = dim;

    return kernel;
}

void GenerateKernel::recordPrepare(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const uint32_t dim,
    Fft &fft, ImageHandle inKernel, const uint32_t nextFrame)
{
    ImageHandle outKernel;
    {
        PROFILER_CPU_SCOPE("  PrepareKernel");

        outKernel = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = Fft::sFftFormat,
                .width = dim,
                .height = dim,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomKernelImageScaled");

        const vk::DescriptorSet descriptorSet = m_preparePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(inKernel).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(outKernel).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            scopeAlloc.child_scope(), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inKernel, gfx::ImageState::ComputeShaderRead},
                    {outKernel, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  PrepareKernel");

        const uvec3 groupCount = m_preparePass.groupCount(uvec3{dim, dim, 1u});
        m_preparePass.record(cb, groupCount, Span{&descriptorSet, 1});
    }

    m_kernelDft = fft.record(
        WHEELS_MOV(scopeAlloc), cb, outKernel, nextFrame, false, "BloomKernel");
    gRenderResources.images->preserve(m_kernelDft);

    gRenderResources.images->release(outKernel);
}

void GenerateKernel::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    if (gRenderResources.images->isValidHandle(m_kernelDft))
        gRenderResources.images->release(m_kernelDft);
}

} // namespace render::bloom
