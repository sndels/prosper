#include "DepthOfFieldFlatten.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sFlattenFactor = 8;

vk::Extent2D getInputExtent(
    const RenderResources &resources, ImageHandle illumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(illumination).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/flatten.comp",
        .debugName = String{alloc, "DepthOfFieldFlattenCS"},
        .groupSize = uvec3{sFlattenFactor, sFlattenFactor, 1u},
    };
}

} // namespace

DepthOfFieldFlatten::DepthOfFieldFlatten(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    WHEELS_ASSERT(_resources != nullptr);
}

void DepthOfFieldFlatten::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldFlatten::Output DepthOfFieldFlatten::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    ImageHandle halfResCircleOfConfusion, const uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D inputExtent =
            getInputExtent(*_resources, halfResCircleOfConfusion);

        ret.tileMinMaxCircleOfConfusion = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = (inputExtent.width - 1) / sFlattenFactor + 1,
                .height = (inputExtent.height - 1) / sFlattenFactor + 1,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "tileMinMaxCircleOfConfusion");

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(halfResCircleOfConfusion)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(ret.tileMinMaxCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {halfResCircleOfConfusion, ImageState::ComputeShaderRead},
                    {ret.tileMinMaxCircleOfConfusion,
                     ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Flatten");

        const uvec3 extent = uvec3{inputExtent.width, inputExtent.height, 1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, extent, Span{&storageSet, 1});
    }

    return ret;
}
