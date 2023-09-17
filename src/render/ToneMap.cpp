#include "ToneMap.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../utils/Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    float exposure{1.f};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle inColor)
{
    const vk::Extent3D extent = resources.images.resource(inColor).extent;
    assert(extent.depth == 1);

    return vk::Extent2D{
        .width = extent.width,
        .height = extent.height,
    };
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/tone_map.comp",
        .debugName = String{alloc, "ToneMapCS"},
    };
}

} // namespace

ToneMap::ToneMap(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    assert(_resources != nullptr);
}

void ToneMap::recompileShaders(ScopedScratch scopeAlloc)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);
}

void ToneMap::drawUi()
{
    ImGui::DragFloat("Exposure", &_exposure, 0.5f, 0.001f, 10000.f);
}

ToneMap::Output ToneMap::record(
    vk::CommandBuffer cb, ImageHandle inColor, const uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent = getRenderExtent(*_resources, inColor);

        ret = createOutputs(renderExtent);

        _computePass.updateDescriptorSet<2>(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images.resource(inColor).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.toneMapped).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
            });

        transition<2>(
            *_resources, cb,
            {
                {inColor, ImageState::ComputeShaderRead},
                {ret.toneMapped, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "ToneMap");

        const uvec3 groups = uvec3{
            (uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u + 1u,
            1u};

        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(
            cb,
            PCBlock{
                .exposure = _exposure,
            },
            groups, Span{&storageSet, 1});
    }

    return ret;
}

ToneMap::Output ToneMap::createOutputs(const vk::Extent2D &size)
{
    return Output{
        .toneMapped = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR8G8B8A8Unorm,
                .width = size.width,
                .height = size.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eSampled |         // Debug
                    vk::ImageUsageFlagBits::eStorage |         // ToneMap
                    vk::ImageUsageFlagBits::eColorAttachment | // ImGui
                    vk::ImageUsageFlagBits::eTransferSrc,      // Blit to swap
                                                               // image
            },
            "toneMapped"),
    };
}