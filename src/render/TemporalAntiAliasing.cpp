#include "TemporalAntiAliasing.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    uint32_t ignoreHistory{0};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle inColor)
{
    const vk::Extent3D extent = resources.images.resource(inColor).extent;
    WHEELS_ASSERT(extent.depth == 1);

    return vk::Extent2D{
        .width = extent.width,
        .height = extent.height,
    };
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/taa_resolve.comp",
        .debugName = String{alloc, "TaaResolveCS"},
    };
}

} // namespace

TemporalAntiAliasing::TemporalAntiAliasing(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    WHEELS_ASSERT(_resources != nullptr);
}

void TemporalAntiAliasing::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDSLayout, 1});
}

TemporalAntiAliasing::Output TemporalAntiAliasing::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inIllumination,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, inIllumination);

        ret = createOutputs(renderExtent);

        vk::Extent3D previousExtent;
        if (_resources->images.isValidHandle(_previousResolveOutput))
            previousExtent =
                _resources->images.resource(_previousResolveOutput).extent;

        // TODO: Reset history from app when camera or scene is toggled,
        // projection changes
        bool ignoreHistory = false;
        const char *resolveDebugName = "previousResolvedIllumination";
        if (renderExtent.width != previousExtent.width ||
            renderExtent.height != previousExtent.height)
        {
            if (_resources->images.isValidHandle(_previousResolveOutput))
                _resources->images.release(_previousResolveOutput);

            // Create dummy texture that won't be read from to satisfy binds
            _previousResolveOutput =
                createIllumination(*_resources, renderExtent, resolveDebugName);
            ignoreHistory = true;
        }
        else // We clear debug names each frame
            _resources->images.appendDebugName(
                _previousResolveOutput, resolveDebugName);

        const StaticArray descriptorInfos{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = _resources->images.resource(inIllumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    _resources->images.resource(_previousResolveOutput).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    _resources->images.resource(ret.resolvedIllumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        };
        _computePass.updateDescriptorSet(
            WHEELS_MOV(scopeAlloc), nextFrame, descriptorInfos);

        transition<3>(
            *_resources, cb,
            {
                {inIllumination, ImageState::ComputeShaderRead},
                {_previousResolveOutput, ImageState::ComputeShaderRead},
                {ret.resolvedIllumination, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "TemporalAntiAliasing");

        const uvec3 groups = uvec3{
            (uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u + 1u,
            1u};

        const vk::DescriptorSet descriptorSet =
            _computePass.storageSet(nextFrame);

        _computePass.record(
            cb,
            PCBlock{
                .ignoreHistory = ignoreHistory,
            },
            groups, Span{&descriptorSet, 1});

        _resources->images.release(_previousResolveOutput);
        _previousResolveOutput = ret.resolvedIllumination;
        _resources->images.preserve(_previousResolveOutput);
    }

    return ret;
}

void TemporalAntiAliasing::releasePreserved()
{
    if (_resources->images.isValidHandle(_previousResolveOutput))
        _resources->images.release(_previousResolveOutput);
}

TemporalAntiAliasing::Output TemporalAntiAliasing::createOutputs(
    const vk::Extent2D &size)
{
    return Output{
        .resolvedIllumination =
            createIllumination(*_resources, size, "ResolvedIllumination"),
    };
}
