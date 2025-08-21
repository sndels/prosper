#include "Separate.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "render/bloom/Fft.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/separate.h>
#include <type_traits>

using namespace glm;
using namespace wheels;

namespace render::bloom
{

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/separate.comp",
        .debugName = String{alloc, "BloomSeparateCS"},
    };
}

uint32_t specializationIndex(ResolutionScale scale)
{
    uint32_t ret = 0;

    ret = static_cast<std::underlying_type_t<ResolutionScale>>(scale);

    return ret;
}

StaticArray<uint32_t, 2> generateSpecializationConstants()
{
    StaticArray<uint32_t, 2> ret;
    for (const ResolutionScale scale :
         {ResolutionScale::Half, ResolutionScale::Quarter})
    {
        const uint32_t constants = static_cast<uint32_t>(scale);
        const uint32_t index = specializationIndex(scale);
        ret[index] = constants;
    }

    return ret;
}

} // namespace

void Separate::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    const StaticArray specializationConstants =
        generateSpecializationConstants();

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        specializationConstants.span());

    m_initialized = true;
}

void Separate::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void Separate::drawUi()
{
    ImGui::SliderFloat("Threshold", &m_threshold, 0.f, 10.f);
}

ImageHandle Separate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    ResolutionScale resolutionScale, Technique technique,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Separate");

    ImageHandle ret;
    {
        const vk::Extent2D inputExtent = getExtent2D(input.illumination);
        const uint32_t u32ResolutionScale =
            bloomResolutionScale(resolutionScale);
        const uint32_t dim = std::max(
            std::bit_ceil(std::max(inputExtent.width, inputExtent.height)) /
                u32ResolutionScale,
            Fft::sMinResolution);

        ret = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = sIlluminationFormat,
                .width = technique == Technique::Fft
                             ? dim
                             : inputExtent.width / u32ResolutionScale,
                .height = technique == Technique::Fft
                              ? dim
                              : inputExtent.height / u32ResolutionScale,
                // Three mips from half or quarter resolution
                .mipCount =
                    technique == Technique::MultiResolutionBlur ? 4u : 1u,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "BloomWorkingImage");

        const vk::DescriptorSet descriptorSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->subresourceViews(ret)[0],
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler =
                        gRenderResources.bilinearBorderTransparentBlackSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {input.illumination,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {ret, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Separate");

        const SeparatePC pcBlock{
            .invInResolution = 1.f /
                               vec2{
                                   static_cast<float>(inputExtent.width),
                                   static_cast<float>(inputExtent.height),
                               },
            .threshold = m_threshold,
        };
        const uvec3 groupCount = m_computePass.groupCount(uvec3{dim, dim, 1u});
        m_computePass.record(
            cb, pcBlock, groupCount, Span{&descriptorSet, 1},
            ComputePassOptionalRecordArgs{
                .specializationIndex = specializationIndex(resolutionScale),
            });
    }

    return ret;
}

} // namespace render::bloom
