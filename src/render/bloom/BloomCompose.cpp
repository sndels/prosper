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

} // namespace

void BloomCompose::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

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
    const uint32_t nextFrame)
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

        m_computePass.updateDescriptorSet(
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
                .images = StaticArray<ImageTransition, 2>{{
                    {input.illumination, ImageState::ComputeShaderRead},
                    {input.bloomHighlights, ImageState::ComputeShaderRead},
                    {ret, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Compose");

        const vk::DescriptorSet descriptorSet =
            m_computePass.storageSet(nextFrame);

        const ComposePC pcBlock
        {
            .invDimSquared = 1.f / (bloomExtent.width * bloomExtent.width);
        };
        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{illuminationExtent.width, illuminationExtent.height, 1u});
        m_computePass.record(cb, pcBlock, groupCount, Span{&descriptorSet, 1});
    }

    return ret;
}
