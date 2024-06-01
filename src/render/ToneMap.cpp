#include "ToneMap.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    float exposure{1.f};
    float contrast{1.f};
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
        .relPath = "shader/tone_map.comp",
        .debugName = String{alloc, "ToneMapCS"},
    };
}

} // namespace

void ToneMap::init(
    ScopedScratch scopeAlloc, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(resources != nullptr);

    _resources = resources;
    _computePass.init(
        scopeAlloc.child_scope(), staticDescriptorsAlloc,
        shaderDefinitionCallback);

    _lut.init(
        WHEELS_MOV(scopeAlloc), resPath("texture/tony_mc_mapface.dds"),
        ImageState::ComputeShaderSampledRead);

    _initialized = true;
}

void ToneMap::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void ToneMap::drawUi()
{
    WHEELS_ASSERT(_initialized);

    ImGui::DragFloat("Exposure", &_exposure, 0.01f, 0.001f, 10000.f);
    ImGui::DragFloat("Contrast", &_contrast, 0.01f, 0.001f, 10000.f);
}

ToneMap::Output ToneMap::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, ImageHandle inColor,
    const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent = getRenderExtent(*_resources, inColor);

        ret = createOutputs(renderExtent);

        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = _resources->images.resource(inColor).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{_lut.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _resources->bilinearSampler,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = _resources->images.resource(ret.toneMapped).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }};
        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inColor, ImageState::ComputeShaderRead},
                    {ret.toneMapped, ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "ToneMap");

        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};

        const vk::DescriptorSet storageSet = _computePass.storageSet(nextFrame);
        _computePass.record(
            cb,
            PCBlock{
                .exposure = _exposure,
                .contrast = _contrast,
            },
            extent, Span{&storageSet, 1});
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
