#include "DepthOfFieldCombine.hpp"

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

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/dof/combine.comp",
        .debugName = String{alloc, "DepthOfFieldCombineCS"},
    };
}

} // namespace

void DepthOfFieldCombine::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!_initialized);

    _computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback);

    _initialized = true;
}

void DepthOfFieldCombine::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldCombine::Output DepthOfFieldCombine::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(input.illumination);

        ret.combinedIlluminationDoF =
            createIllumination(renderExtent, "CombinedIllumnationDoF");

        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResFgBokehWeight)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResBgBokehWeight)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.halfResCircleOfConfusion)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.illumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(ret.combinedIlluminationDoF)
                                 .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }};
        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 5>{{
                    {input.halfResFgBokehWeight, ImageState::ComputeShaderRead},
                    {input.halfResBgBokehWeight, ImageState::ComputeShaderRead},
                    {input.halfResCircleOfConfusion,
                     ImageState::ComputeShaderRead},
                    {input.illumination, ImageState::ComputeShaderRead},
                    {ret.combinedIlluminationDoF,
                     ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Combine");

        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, extent, Span{&storageSet, 1});
    }

    return ret;
}
