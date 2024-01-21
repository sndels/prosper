#include "DepthOfFieldSetup.hpp"

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

enum BindingSet : uint32_t
{
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    float focusDistance{0.f};
    float maxBackgroundCoC{0.f};
};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, ImageHandle illumination)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(illumination).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width / 2,
        .height = targetExtent.height / 2,
    };
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 48;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/dof/setup.comp",
        .debugName = String{alloc, "DepthOfFieldSetupCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

DepthOfFieldSetup::DepthOfFieldSetup(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDsLayout)
: _resources{resources}
, _computePass{WHEELS_MOV(scopeAlloc), device,
               staticDescriptorsAlloc, shaderDefinitionCallback,
               StorageBindingSet,      Span{&camDsLayout, 1}}
{
    WHEELS_ASSERT(_resources != nullptr);
}

void DepthOfFieldSetup::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDsLayout)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDsLayout, 1});
}

DepthOfFieldSetup::Output DepthOfFieldSetup::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const Input &input, const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

        ret.halfResIllumination = createIllumination(
            *_resources, renderExtent, "HalfResIllumination");
        ret.halfResCircleOfConfusion = _resources->images.create(
            ImageDescription{
                .format = vk::Format::eR16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "HalfResCircleOfConfusion");

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.illumination).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images.resource(input.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.halfResIllumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(ret.halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = _resources->nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {input.illumination, ImageState::ComputeShaderRead},
                    {input.depth, ImageState::ComputeShaderRead},
                    {ret.halfResIllumination, ImageState::ComputeShaderWrite},
                    {ret.halfResCircleOfConfusion,
                     ImageState::ComputeShaderWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Setup");

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const uint32_t cameraOffset = cam.bufferOffset();

        const CameraParameters &camParams = cam.parameters();
        const float maxBgCoCInUnits =
            (camParams.apertureDiameter * camParams.focalLength) /
            (camParams.focusDistance - camParams.focalLength);

        const float maxBgCoCInHalfResPixels =
            (maxBgCoCInUnits / cam.sensorWidth()) * renderExtent.width;

        const PCBlock pcBlock{
            .focusDistance = camParams.focusDistance,
            .maxBackgroundCoC = maxBgCoCInHalfResPixels,
        };
        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};
        _computePass.record(
            cb, pcBlock, extent, descriptorSets, Span{&cameraOffset, 1});
    }

    return ret;
}
