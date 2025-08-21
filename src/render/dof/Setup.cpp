#include "Setup.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "render/dof/DepthOfField.hpp"
#include "scene/Camera.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/dof/setup.h>

using namespace glm;
using namespace wheels;

namespace render::dof
{

namespace
{

enum BindingSet : uint8_t
{
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
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

void Setup::init(ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = Span{&camDsLayout, 1},
        });

    m_initialized = true;
}

void Setup::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDsLayout, 1});
}

Setup::Output Setup::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::Camera &cam,
    const Input &input, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Setup");

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRoundedUpHalfExtent2D(input.illumination);

        const uint32_t mipCount =
            getMipCount(std::max(renderExtent.width, renderExtent.height));
        ret.halfResIllumination = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = sIlluminationFormat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .mipCount = mipCount,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,

            },
            "HalfResIllumination");

        ret.halfResCircleOfConfusion = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = vk::Format::eR16Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
            "HalfResCircleOfConfusion");

        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.halfResIllumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(ret.halfResCircleOfConfusion)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {input.illumination, gfx::ImageState::ComputeShaderRead},
                    {input.depth, gfx::ImageState::ComputeShaderRead},
                    {ret.halfResIllumination,
                     gfx::ImageState::ComputeShaderWrite},
                    {ret.halfResCircleOfConfusion,
                     gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Setup");

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = storageSet;

        const uint32_t cameraOffset = cam.bufferOffset();

        const scene::CameraParameters &camParams = cam.parameters();
        const float maxBgCoCInUnits =
            (camParams.apertureDiameter * camParams.focalLength) /
            (camParams.focusDistance - camParams.focalLength);

        // Similar math is also in Dilate
        const float maxBgCoCInHalfResPixels =
            (maxBgCoCInUnits / cam.sensorWidth()) *
            static_cast<float>(renderExtent.width);

        const SetupPC pcBlock{
            .focusDistance = camParams.focusDistance,
            .maxBackgroundCoC = maxBgCoCInHalfResPixels,
            .maxCoC = maxBgCoCInHalfResPixels * DepthOfField::sMaxFgCoCFactor,
        };
        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});
        m_computePass.record(
            cb, pcBlock, groupCount, descriptorSets,
            ComputePassOptionalRecordArgs{
                .dynamicOffsets = Span{&cameraOffset, 1},
            });
    }

    return ret;
}

} // namespace render::dof
