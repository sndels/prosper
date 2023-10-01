#include "DepthOfFieldGather.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    uint frameIndex{0};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle halfResIllumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(halfResIllumination).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

ComputePass::Shader backgroundDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(defines, "GATHER_BACKGROUND");
    assert(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/dof/gather.comp",
        .debugName = String{alloc, "DepthOfFieldGatherBgCS"},
        .defines = WHEELS_MOV(defines),
    };
}

ComputePass::Shader foregroundDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/gather.comp",
        .debugName = String{alloc, "DepthOfFieldGatherFgCS"},
    };
}

} // namespace

DepthOfFieldGather::DepthOfFieldGather(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _backgroundPass{scopeAlloc.child_scope(), device, staticDescriptorsAlloc, backgroundDefinitionCallback}
, _foregroundPass{
      scopeAlloc.child_scope(), device, staticDescriptorsAlloc,
      foregroundDefinitionCallback}
{
    assert(_resources != nullptr);
}

void DepthOfFieldGather::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    _backgroundPass.recompileShader(
        scopeAlloc.child_scope(), backgroundDefinitionCallback);
    _foregroundPass.recompileShader(
        scopeAlloc.child_scope(), foregroundDefinitionCallback);
}

DepthOfFieldGather::Output DepthOfFieldGather::record(
    vk::CommandBuffer cb, const Input &input, GatherType gatherType,
    const uint32_t nextFrame, Profiler *profiler)
{
    assert(profiler != nullptr);
    assert(gatherType < GatherType_Count);

    ComputePass *computePass = gatherType == GatherType_Foreground
                                   ? &_foregroundPass
                                   : &_backgroundPass;

    if (gatherType == GatherType_Foreground)
        _frameIndex = (_frameIndex + 1) % 128;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.halfResIllumination);

        ret.halfResBokehColorWeight = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            gatherType == GatherType_Background ? "halfResBgBokehColorWeight"
                                                : "halfResFgBokehColorWeight");

        computePass->updateDescriptorSet(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.halfResIllumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.halfResCoC).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.dilatedTileMinMaxCoC)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.halfResBokehColorWeight)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            });

        transition<4>(
            *_resources, cb,
            {
                {input.halfResIllumination, ImageState::ComputeShaderRead},
                {input.halfResCoC, ImageState::ComputeShaderRead},
                {input.dilatedTileMinMaxCoC, ImageState::ComputeShaderRead},
                {ret.halfResBokehColorWeight, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(
            cb, gatherType == GatherType_Background ? "  GatherBackground"
                                                    : "  GatherForeground");

        const PCBlock pcBlock{
            .frameIndex = _frameIndex,
        };
        const uvec3 groups = uvec3{
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
                1u,
            1u};
        const vk::DescriptorSet storageSet = computePass->storageSet(nextFrame);
        computePass->record(cb, pcBlock, groups, Span{&storageSet, 1});
    }

    return ret;
}
