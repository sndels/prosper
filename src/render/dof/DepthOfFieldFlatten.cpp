#include "DepthOfFieldFlatten.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/flatten.comp",
        .debugName = String{alloc, "DepthOfFieldFlattenCS"},
        .groupSize =
            uvec3{
                DepthOfFieldFlatten::sFlattenFactor,
                DepthOfFieldFlatten::sFlattenFactor, 1u},
    };
}

} // namespace

void DepthOfFieldFlatten::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback);

    m_initialized = true;
}

void DepthOfFieldFlatten::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldFlatten::Output DepthOfFieldFlatten::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle halfResCircleOfConfusion, const uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE(profiler, "  Flatten");

    Output ret;
    {
        const vk::Extent2D inputExtent = getExtent2D(halfResCircleOfConfusion);

        ret.tileMinMaxCircleOfConfusion = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = roundedUpQuotient(
                    inputExtent.width, DepthOfFieldFlatten::sFlattenFactor),
                .height = roundedUpQuotient(
                    inputExtent.height, DepthOfFieldFlatten::sFlattenFactor),
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "tileMinMaxCircleOfConfusion");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.tileMinMaxCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {halfResCircleOfConfusion, ImageState::ComputeShaderRead},
                    {ret.tileMinMaxCircleOfConfusion,
                     ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(profiler, cb, "  Flatten");

        const uvec3 extent = uvec3{inputExtent.width, inputExtent.height, 1u};
        const vk::DescriptorSet storageSet =
            m_computePass.storageSet(nextFrame);
        m_computePass.record(cb, extent, Span{&storageSet, 1});
    }

    return ret;
}
