#include "Convolution.hpp"

#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/convolution.h>

using namespace glm;
using namespace wheels;

namespace render::bloom
{

namespace
{
constexpr uvec3 sGroupSize{16, 16, 1};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/convolution.comp",
        .debugName = String{alloc, "BloomConvolutionCS"},
        .groupSize = sGroupSize,
    };
}

} // namespace

void Convolution::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    m_initialized = true;
}

void Convolution::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void Convolution::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const InputOutput &inputOutput, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Convolution");

    const vk::Extent2D highlightsExtent =
        getExtent2D(inputOutput.inOutHighlightsDft);
    const vk::Extent2D kernelExtent = getExtent2D(inputOutput.inKernelDft);
    WHEELS_ASSERT(highlightsExtent.width == highlightsExtent.height);
    WHEELS_ASSERT(kernelExtent.width == kernelExtent.height);
    WHEELS_ASSERT(highlightsExtent.width == kernelExtent.width);
    WHEELS_ASSERT(highlightsExtent.width % sGroupSize.x == 0);
    WHEELS_ASSERT(highlightsExtent.height % sGroupSize.y == 0);

    const vk::DescriptorSet descriptorSet = m_computePass.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(inputOutput.inOutHighlightsDft)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(inputOutput.inKernelDft)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 2>{{
                {inputOutput.inOutHighlightsDft,
                 gfx::ImageState::ComputeShaderReadWrite},
                {inputOutput.inKernelDft, gfx::ImageState::ComputeShaderRead},
            }},
        });

    PROFILER_GPU_SCOPE(cb, "  Convolution");

    const ConvolutionPC pcBlock{
        .scale = inputOutput.convolutionScale,
    };

    const uvec3 groupCount = m_computePass.groupCount(
        uvec3{highlightsExtent.width, highlightsExtent.height, 1u});
    m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
}

} // namespace render::bloom
