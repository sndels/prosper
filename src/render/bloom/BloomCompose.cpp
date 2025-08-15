#include "BloomCompose.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/compose.h>

using namespace glm;
using namespace wheels;

namespace render::bloom
{

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/compose.comp",
        .debugName = String{alloc, "BloomComposeCS"},
    };
}

struct BloomComposeConstants
{
    VkBool32 sampleBiquadratic{VK_FALSE};
    VkBool32 multiResolution{VK_FALSE};
};

uint32_t specializationIndex(const BloomComposeConstants &constants)
{
    uint32_t ret = 0;

    ret = static_cast<uint32_t>(constants.sampleBiquadratic);
    ret |= static_cast<uint32_t>(constants.multiResolution) << 1;

    return ret;
}

StaticArray<BloomComposeConstants, 4> generateSpecializationConstants()
{
    StaticArray<BloomComposeConstants, 4> ret;
    for (const VkBool32 sampleBiquadratic : {VK_FALSE, VK_TRUE})
    {
        for (const BloomTechnique technique :
             {BloomTechnique::Fft, BloomTechnique::MultiResolutionBlur})
        {
            const BloomComposeConstants constants{
                .sampleBiquadratic = sampleBiquadratic,
                .multiResolution =
                    technique == BloomTechnique::MultiResolutionBlur ? VK_TRUE
                                                                     : VK_FALSE,
            };
            const uint32_t index = specializationIndex(constants);
            ret[index] = constants;
        }
    }

    return ret;
}

} // namespace

void BloomCompose::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    const StaticArray specializationConstants =
        generateSpecializationConstants();

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        specializationConstants.span());

    m_initialized = true;
}

void BloomCompose::drawUi(BloomTechnique technique)
{
    if (technique == BloomTechnique::MultiResolutionBlur)
        ImGui::DragFloat3(
            "Blend factors", &m_blendFactors[0], .01f, .00f, 2.f, "%.2f");
    ImGui::Checkbox("Biquadratic sampling", &m_biquadraticSampling);
}

void BloomCompose::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

ImageHandle BloomCompose::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    BloomResolutionScale resolutionScale, BloomTechnique technique,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Compose");

    ImageHandle ret;
    {
        const vk::Extent2D illuminationExtent = getExtent2D(input.illumination);
        const vk::Extent2D bloomExtent = getExtent2D(input.bloomHighlights);

        ret = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = sIlluminationFormat,
                .width = illuminationExtent.width,
                .height = illuminationExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "IlluminationWithBloom");

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
                        gRenderResources.images->resource(input.bloomHighlights)
                            .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images->resource(ret).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.bilinearSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 3>{{
                    {input.illumination,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {input.bloomHighlights,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {ret, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Compose");

        const BloomComposeConstants constants{
            .sampleBiquadratic = m_biquadraticSampling ? VK_TRUE : VK_FALSE,
            .multiResolution = technique == BloomTechnique::MultiResolutionBlur
                                   ? VK_TRUE
                                   : VK_FALSE,
        };
        const ComposePC pcBlock{
            .illuminationResolution =
                vec2(illuminationExtent.width, illuminationExtent.height),
            .invIlluminationResolution =
                1.f / vec2(illuminationExtent.width, illuminationExtent.height),
            .blendFactors = m_blendFactors,
            .invBloomDimSquared =
                1.f / static_cast<float>(bloomExtent.width * bloomExtent.width),
            .resolutionScale = bloomResolutionScale(resolutionScale),
        };
        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{illuminationExtent.width, illuminationExtent.height, 1u});
        m_computePass.record(
            cb, pcBlock, groupCount, Span{&descriptorSet, 1},
            ComputePassOptionalRecordArgs{
                .specializationIndex = specializationIndex(constants),
            });
    }

    return ret;
}

} // namespace render::bloom
