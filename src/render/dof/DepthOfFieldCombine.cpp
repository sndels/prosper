#include "DepthOfFieldCombine.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle illumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(illumination).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/combine.comp",
        .debugName = String{alloc, "DepthOfFieldCombineCS"},
    };
}

} // namespace

DepthOfFieldCombine::DepthOfFieldCombine(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    assert(_resources != nullptr);
}

void DepthOfFieldCombine::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);
}

DepthOfFieldCombine::Output DepthOfFieldCombine::record(
    vk::CommandBuffer cb, const Input &input, const uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

        ret.combinedIlluminationDoF = createIllumination(
            *_resources, renderExtent, "CombinedIllumnationDoF");

        _computePass.updateDescriptorSet<5>(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.halfResFgBokehWeight)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.halfResBgBokehWeight)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(input.halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.illumination).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.combinedIlluminationDoF)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            });

        transition<5>(
            *_resources, cb,
            {
                {input.halfResFgBokehWeight, ImageState::ComputeShaderRead},
                {input.halfResBgBokehWeight, ImageState::ComputeShaderRead},
                {input.halfResCircleOfConfusion, ImageState::ComputeShaderRead},
                {input.illumination, ImageState::ComputeShaderRead},
                {ret.combinedIlluminationDoF, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Combine");

        const uvec3 groups = uvec3{
            (uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u + 1u,
            1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, groups, Span{&storageSet, 1});
    }

    return ret;
}
