#include "BloomCompose.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/bloom/compose.h>

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/bloom/compose.comp",
        .debugName = String{alloc, "BloomComposeCS"},
    };
}

uint32_t specializationIndex(bool sampleBiquadratic)
{
    uint32_t ret = 0;

    ret = static_cast<uint32_t>(sampleBiquadratic);

    return ret;
}

StaticArray<uint32_t, 2> generateSpecializationConstants()
{
    StaticArray<uint32_t, 2> ret;
    for (const bool sampleBiquadratic : {false, true})
    {
        const uint32_t constants = static_cast<uint32_t>(sampleBiquadratic);
        const uint32_t index = specializationIndex(sampleBiquadratic);
        ret[index] = constants;
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
    BloomResolutionScale resolutionScale, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Compose");

    ImageHandle ret;
    {
        const vk::Extent2D illuminationExtent = getExtent2D(input.illumination);
        const vk::Extent2D bloomExtent = getExtent2D(input.bloomHighlights);
        WHEELS_ASSERT(bloomExtent.width == bloomExtent.height);

        ret = gRenderResources.images->create(
            ImageDescription{
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
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.bloomHighlights)
                            .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images->resource(ret).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.bilinearSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 3>{{
                    {input.illumination, ImageState::ComputeShaderSampledRead},
                    {input.bloomHighlights,
                     ImageState::ComputeShaderSampledRead},
                    {ret, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Compose");

        const bool sampleBiquadratic =
            resolutionScale != BloomResolutionScale::Half;

        const ComposePC pcBlock{
            .illuminationResolution =
                vec2(illuminationExtent.width, illuminationExtent.height),
            .invIlluminationResolution =
                1.f / vec2(illuminationExtent.width, illuminationExtent.height),
            .invBloomDimSquared =
                1.f / static_cast<float>(bloomExtent.width * bloomExtent.width),
            .resolutionScale = bloomResolutionScale(resolutionScale),
        };
        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{illuminationExtent.width, illuminationExtent.height, 1u});
        m_computePass.record(
            cb, pcBlock, groupCount, Span{&descriptorSet, 1},
            ComputePassOptionalRecordArgs{
                .specializationIndex = specializationIndex(sampleBiquadratic),
            });
    }

    return ret;
}
