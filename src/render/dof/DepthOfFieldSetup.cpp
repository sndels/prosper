#include "DepthOfFieldSetup.hpp"

#include <imgui.h>

#include "../../gfx/VkUtils.hpp"
#include "../../scene/Camera.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../RenderResources.hpp"
#include "../RenderTargets.hpp"
#include "../Utils.hpp"
#include "DepthOfField.hpp"

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
    float maxCoC{0.f};
};

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

void DepthOfFieldSetup::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        shaderDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = Span{&camDsLayout, 1},
        });

    m_initialized = true;
}

void DepthOfFieldSetup::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDsLayout, 1});
}

DepthOfFieldSetup::Output DepthOfFieldSetup::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const Input &input, const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE(profiler, "  Setup");

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRoundedUpHalfExtent2D(input.illumination);

        const uint32_t mipCount =
            static_cast<uint32_t>(floor(log2(static_cast<float>(
                std::max(renderExtent.width, renderExtent.height))))) +
            1;
        ret.halfResIllumination = gRenderResources.images->create(
            ImageDescription{
                .format = sIlluminationFormat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .mipCount = mipCount,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "HalfResIllumination");

        ret.halfResCircleOfConfusion = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "HalfResCircleOfConfusion");

        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.halfResIllumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {input.illumination, ImageState::ComputeShaderRead},
                    {input.depth, ImageState::ComputeShaderRead},
                    {ret.halfResIllumination, ImageState::ComputeShaderWrite},
                    {ret.halfResCircleOfConfusion,
                     ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(profiler, cb, "  Setup");

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = m_computePass.storageSet(nextFrame);

        const uint32_t cameraOffset = cam.bufferOffset();

        const CameraParameters &camParams = cam.parameters();
        const float maxBgCoCInUnits =
            (camParams.apertureDiameter * camParams.focalLength) /
            (camParams.focusDistance - camParams.focalLength);

        // Similar math is also in Dilate
        const float maxBgCoCInHalfResPixels =
            (maxBgCoCInUnits / cam.sensorWidth()) *
            static_cast<float>(renderExtent.width);

        const PCBlock pcBlock{
            .focusDistance = camParams.focusDistance,
            .maxBackgroundCoC = maxBgCoCInHalfResPixels,
            .maxCoC = maxBgCoCInHalfResPixels * DepthOfField::sMaxFgCoCFactor,
        };
        const uvec3 extent = uvec3{renderExtent.width, renderExtent.height, 1u};
        m_computePass.record(
            cb, pcBlock, extent, descriptorSets, Span{&cameraOffset, 1});
    }

    return ret;
}
