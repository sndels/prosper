#include "DepthOfFieldCombine.hpp"

#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../scene/Camera.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"
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
    WHEELS_ASSERT(targetExtent.depth == 1);

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
    WHEELS_ASSERT(_resources != nullptr);
}

void DepthOfFieldCombine::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldCombine::Output DepthOfFieldCombine::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

        ret.combinedIlluminationDoF = createIllumination(
            *_resources, renderExtent, "CombinedIllumnationDoF");

        const StaticArray descriptorInfos{{
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
                .imageView =
                    _resources->images.resource(input.halfResCircleOfConfusion)
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
        }};
        _computePass.updateDescriptorSet(
            WHEELS_MOV(scopeAlloc), nextFrame, descriptorInfos);

        transition<5>(
            *_resources, cb,
            {{
                {input.halfResFgBokehWeight, ImageState::ComputeShaderRead},
                {input.halfResBgBokehWeight, ImageState::ComputeShaderRead},
                {input.halfResCircleOfConfusion, ImageState::ComputeShaderRead},
                {input.illumination, ImageState::ComputeShaderRead},
                {ret.combinedIlluminationDoF, ImageState::ComputeShaderWrite},
            }});

        const auto _s = profiler->createCpuGpuScope(cb, "  Combine");

        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, extent, Span{&storageSet, 1});
    }

    return ret;
}
