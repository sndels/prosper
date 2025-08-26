#include "Accumulate.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace render::svgf
{

namespace
{

enum BindingSet : uint8_t
{
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct AccumulateConstants
{
    VkBool32 ignoreHistory{VK_FALSE};
};

uint32_t specializationIndex(AccumulateConstants constants)
{
    uint32_t ret = 0;

    ret |= (uint32_t)constants.ignoreHistory;

    return ret;
}

Array<AccumulateConstants> generateSpecializationConstants(Allocator &alloc)
{
    Array<AccumulateConstants> ret{alloc};
    ret.resize((1 << 7) - 1);
    for (const VkBool32 ignoreHistory : {VK_FALSE, VK_TRUE})
    {

        const AccumulateConstants constants{
            .ignoreHistory = ignoreHistory,
        };
        const uint32_t index = specializationIndex(constants);

        ret[index] = constants;
    }

    return ret;
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{

    const size_t len = 256;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/svgf/accumulate.comp",
        .debugName = String{alloc, "SvgfAccumulateCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

void Accumulate::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    const Array<AccumulateConstants> specializationConstants =
        generateSpecializationConstants(scopeAlloc);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        specializationConstants.span(),
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = Span{&camDsLayout, 1},
        });

    m_initialized = true;
}

bool Accumulate::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(m_initialized);

    return m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDSLayout, 1});
}

Accumulate::Output Accumulate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::Camera &cam,
    const Input &input, bool ignoreHistory, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Accumulate");

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(input.color);

        ret = Output{
            .color = gRenderResources.images->create(
                gfx::ImageDescription{
                    .format = vk::Format::eR16G16B16A16Sfloat,
                    .width = renderExtent.width,
                    .height = renderExtent.height,
                    .usageFlags = vk::ImageUsageFlagBits::eStorage |
                                  vk::ImageUsageFlagBits::eSampled,
                },
                "SvgfIntegratedColor"),
            .moments = gRenderResources.images->create(
                gfx::ImageDescription{
                    // TODO: Is 32bit overkill?
                    .format = vk::Format::eR32G32Sfloat,
                    .width = renderExtent.width,
                    .height = renderExtent.height,
                    .usageFlags = vk::ImageUsageFlagBits::eStorage |
                                  vk::ImageUsageFlagBits::eSampled,
                },
                "SvgfIntegratedMoments"),
        };

        vk::Extent3D previousExtent;
        if (gRenderResources.images->isValidHandle(m_previousIntegratedMoments))
            previousExtent =
                gRenderResources.images->resource(m_previousIntegratedMoments)
                    .extent;

        const char *previousColorDebugName = "SvgfPreviousIntegratedColor";
        const char *previousMomentsDebugName = "SvgfPreviousIntegratedMoments";
        if (ignoreHistory || renderExtent.width != previousExtent.width ||
            renderExtent.height != previousExtent.height)
        {
            if (gRenderResources.images->isValidHandle(
                    m_previousIntegratedMoments))
                gRenderResources.images->release(m_previousIntegratedMoments);
            if (gRenderResources.images->isValidHandle(
                    m_previousIntegratedColor))
                gRenderResources.images->release(m_previousIntegratedColor);

            // Create dummy texture to satisfy binds even though it won't be
            // read from
            m_previousIntegratedColor =
                createIllumination(renderExtent, previousColorDebugName);
            m_previousIntegratedMoments =
                createIllumination(renderExtent, previousMomentsDebugName);
            ignoreHistory = true;
        }
        else
        {
            // We clear debug names each frame
            gRenderResources.images->appendDebugName(
                m_previousIntegratedColor, previousColorDebugName);
            gRenderResources.images->appendDebugName(
                m_previousIntegratedMoments, previousMomentsDebugName);
        }

        ImageHandle previousAlbedoRoughness =
            input.previous_gbuffer.albedoRoughness;
        ImageHandle previousNormalMetallic =
            input.previous_gbuffer.normalMetallic;
        ImageHandle previousDepth = input.previous_gbuffer.depth;
        if (previousAlbedoRoughness.isValid())
            previousAlbedoRoughness = input.gbuffer.albedoRoughness;
        if (previousNormalMetallic.isValid())
            previousNormalMetallic = input.gbuffer.normalMetallic;
        if (previousDepth.isValid())
            previousDepth = input.gbuffer.depth;
        const StaticArray descriptorInfos{{
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.color).view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.gbuffer.albedoRoughness)
                                 .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(input.gbuffer.normalMetallic)
                                 .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.gbuffer.velocity)
                        .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.gbuffer.depth).view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(previousAlbedoRoughness)
                        .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(previousNormalMetallic)
                        .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(previousDepth).view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(m_previousIntegratedColor)
                        .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images
                                 ->resource(m_previousIntegratedMoments)
                                 .view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = gRenderResources.images->resource(ret.color).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(ret.moments).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestSampler,
            }},
        }};
        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 12>{{
                    {input.color, gfx::ImageState::ComputeShaderSampledRead},
                    {input.gbuffer.albedoRoughness,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {input.gbuffer.normalMetallic,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {input.gbuffer.velocity,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {input.gbuffer.depth,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {previousAlbedoRoughness,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {previousNormalMetallic,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {previousDepth, gfx::ImageState::ComputeShaderSampledRead},
                    {m_previousIntegratedColor,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {m_previousIntegratedMoments,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {ret.color, gfx::ImageState::ComputeShaderWrite},
                    {ret.moments, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Accumulate");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = storageSet;

        const uint32_t camOffset = cam.bufferOffset();

        const AccumulateConstants constants{
            .ignoreHistory = ignoreHistory ? VK_TRUE : VK_FALSE,
        };

        m_computePass.record(
            cb, groupCount, descriptorSets,
            ComputePassOptionalRecordArgs{
                .dynamicOffsets = Span{&camOffset, 1},
                .specializationIndex = specializationIndex(constants),
            });

        gRenderResources.images->release(m_previousIntegratedColor);
        gRenderResources.images->release(m_previousIntegratedMoments);
        m_previousIntegratedColor = ret.color;
        m_previousIntegratedMoments = ret.moments;
        gRenderResources.images->preserve(m_previousIntegratedColor);
        gRenderResources.images->preserve(m_previousIntegratedMoments);
    }

    return ret;
}

void Accumulate::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    if (gRenderResources.images->isValidHandle(m_previousIntegratedColor))
        gRenderResources.images->release(m_previousIntegratedColor);
    if (gRenderResources.images->isValidHandle(m_previousIntegratedMoments))
        gRenderResources.images->release(m_previousIntegratedMoments);
}

} // namespace render::svgf
