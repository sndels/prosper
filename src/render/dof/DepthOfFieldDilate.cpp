#include "DepthOfFieldDilate.hpp"

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
        .relPath = "shader/dof/dilate.comp",
        .debugName = String{alloc, "DepthOfFieldDilateCS"},
    };
}

} // namespace

DepthOfFieldDilate::DepthOfFieldDilate(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    WHEELS_ASSERT(_resources != nullptr);
}

void DepthOfFieldDilate::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

DepthOfFieldDilate::Output DepthOfFieldDilate::record(
    vk::CommandBuffer cb, ImageHandle tileMinMaxCoC, const uint32_t nextFrame,
    Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D inputExtent =
            getInputExtent(*_resources, tileMinMaxCoC);

        ret.dilatedTileMinMaxCoC = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16G16Sfloat,
                .width = inputExtent.width,
                .height = inputExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "dilatedTileMinMaxCoC");

        _computePass.updateDescriptorSet(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(tileMinMaxCoC).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.dilatedTileMinMaxCoC)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            });

        transition<2>(
            *_resources, cb,
            {
                {tileMinMaxCoC, ImageState::ComputeShaderRead},
                {ret.dilatedTileMinMaxCoC, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Dilate");

        const uvec3 groups = uvec3{
            (uvec2{inputExtent.width, inputExtent.height} - 1u) / 16u + 1u, 1u};
        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(cb, groups, Span{&storageSet, 1});
    }

    return ret;
}
