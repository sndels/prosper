#include "DepthOfFieldGather.hpp"

#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../scene/Camera.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"
#include "../RenderTargets.hpp"
#include "../Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    ivec2 halfResolution{};
    vec2 invHalfResolution{};
    uint frameIndex{0};
};

ComputePass::Shader backgroundDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(defines, "GATHER_BACKGROUND");
    WHEELS_ASSERT(defines.size() <= len);

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

void DepthOfFieldGather::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!_initialized);

    _backgroundPass.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        backgroundDefinitionCallback);
    _foregroundPass.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        foregroundDefinitionCallback);

    _initialized = true;
}

void DepthOfFieldGather::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(_initialized);

    _backgroundPass.recompileShader(
        scopeAlloc.child_scope(), changedFiles, backgroundDefinitionCallback);
    _foregroundPass.recompileShader(
        scopeAlloc.child_scope(), changedFiles, foregroundDefinitionCallback);
}

DepthOfFieldGather::Output DepthOfFieldGather::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    GatherType gatherType, const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);
    WHEELS_ASSERT(gatherType < GatherType_Count);

    ComputePass *computePass = gatherType == GatherType_Foreground
                                   ? &_foregroundPass
                                   : &_backgroundPass;

    if (gatherType == GatherType_Foreground)
        _frameIndex = (_frameIndex + 1) % 128;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getExtent2D(input.halfResIllumination);

        ret.halfResBokehColorWeight = gRenderResources.images->create(
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
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.halfResIllumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.halfResCoC)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.dilatedTileMinMaxCoC)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.halfResBokehColorWeight)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.trilinearSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {input.halfResIllumination, ImageState::ComputeShaderRead},
                    {input.halfResCoC, ImageState::ComputeShaderRead},
                    {input.dilatedTileMinMaxCoC, ImageState::ComputeShaderRead},
                    {ret.halfResBokehColorWeight,
                     ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(
            cb, gatherType == GatherType_Background ? "  GatherBackground"
                                                    : "  GatherForeground");

        const PCBlock pcBlock{
            .halfResolution =
                ivec2{
                    asserted_cast<int32_t>(renderExtent.width),
                    asserted_cast<int32_t>(renderExtent.height)},
            .invHalfResolution = 1.f /
                                 vec2{
                                     static_cast<float>(renderExtent.width),
                                     static_cast<float>(renderExtent.height)},
            .frameIndex = _frameIndex,
        };
        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};
        const vk::DescriptorSet storageSet = computePass->storageSet(nextFrame);
        computePass->record(cb, pcBlock, extent, Span{&storageSet, 1});
    }

    return ret;
}
