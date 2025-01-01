#include "TemporalAntiAliasing.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "utils/Profiler.hpp"
#include "utils/Ui.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace
{

constexpr StaticArray<
    const char *,
    static_cast<size_t>(TemporalAntiAliasing::ColorClippingType::Count)>
    sColorClippingTypeNames{{COLOR_CLIPPING_TYPE_STRS}};
constexpr StaticArray<
    const char *,
    static_cast<size_t>(TemporalAntiAliasing::VelocitySamplingType::Count)>
    sVelocitySamplingTypeNames{{VELOCITY_SAMPLING_TYPE_STRS}};

enum BindingSet : uint8_t
{
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct TaaResolveConstants
{
    VkBool32 ignoreHistory{VK_FALSE};
    VkBool32 catmullRom{VK_FALSE};
    TemporalAntiAliasing::ColorClippingType colorClipping{
        TemporalAntiAliasing::ColorClippingType::None};
    TemporalAntiAliasing::VelocitySamplingType velocitySampling{
        TemporalAntiAliasing::VelocitySamplingType::Center};
    VkBool32 luminanceWeighting{VK_FALSE};
};

uint32_t specializationIndex(TaaResolveConstants constants)
{
    uint32_t ret = 0;

    ret |= (uint32_t)constants.ignoreHistory;
    ret |= (uint32_t)constants.catmullRom << 1;
    ret |= (uint32_t)constants.colorClipping << 2;
    static_assert(
        (uint32_t)TemporalAntiAliasing::ColorClippingType::Count - 1 < 0b11);
    ret |= (uint32_t)constants.velocitySampling << 4;
    static_assert(
        (uint32_t)TemporalAntiAliasing::VelocitySamplingType::Count - 1 < 0b11);
    ret |= (uint32_t)constants.luminanceWeighting << 6;

    return ret;
}

Array<TaaResolveConstants> generateSpecializationConstants(Allocator &alloc)
{
    Array<TaaResolveConstants> ret{alloc};
    ret.resize((1 << 7) - 1);
    for (const VkBool32 ignoreHistory : {VK_FALSE, VK_TRUE})
    {

        for (const VkBool32 catmullRom : {VK_FALSE, VK_TRUE})
        {
            for (const TemporalAntiAliasing::ColorClippingType colorClipping : {
                     TemporalAntiAliasing::ColorClippingType::MinMax,
                     TemporalAntiAliasing::ColorClippingType::None,
                     TemporalAntiAliasing::ColorClippingType::Variance,
                 })
            {
                for (const TemporalAntiAliasing::VelocitySamplingType
                         velocitySampling :
                     {
                         TemporalAntiAliasing::VelocitySamplingType::Center,
                         TemporalAntiAliasing::VelocitySamplingType::Largest,
                         TemporalAntiAliasing::VelocitySamplingType::Closest,
                     })
                {
                    for (const VkBool32 luminanceWeighting :
                         {VK_FALSE, VK_TRUE})
                    {
                        const TaaResolveConstants constants{
                            .ignoreHistory = ignoreHistory,
                            .catmullRom = catmullRom,
                            .colorClipping = colorClipping,
                            .velocitySampling = velocitySampling,
                            .luminanceWeighting = luminanceWeighting,
                        };
                        const uint32_t index = specializationIndex(constants);

                        ret[index] = constants;
                    }
                }
            }
        }
    }

    return ret;
}

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{

    const size_t len = 256;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendEnumVariantsAsDefines(
        defines, "ColorClipping",
        Span{sColorClippingTypeNames.data(), sColorClippingTypeNames.size()});
    appendEnumVariantsAsDefines(
        defines, "VelocitySampling",
        Span{
            sVelocitySamplingTypeNames.data(),
            sVelocitySamplingTypeNames.size()});
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/taa_resolve.comp",
        .debugName = String{alloc, "TaaResolveCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

void TemporalAntiAliasing::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    const Array<TaaResolveConstants> specializationConstants =
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

void TemporalAntiAliasing::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDSLayout, 1});
}

void TemporalAntiAliasing::drawUi()
{
    WHEELS_ASSERT(m_initialized);

    enumDropdown("Color clipping", m_colorClipping, sColorClippingTypeNames);
    enumDropdown(
        "Velocity sampling", m_velocitySampling, sVelocitySamplingTypeNames);

    ImGui::Checkbox("Catmull-Rom history samples", &m_catmullRom);
    ImGui::Checkbox("Luminance Weighting", &m_luminanceWeighting);
}

TemporalAntiAliasing::Output TemporalAntiAliasing::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const Input &input, const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("TemporalAntiAliasing");

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(input.illumination);

        ret = Output{
            .resolvedIllumination =
                createIllumination(renderExtent, "ResolvedIllumination"),
        };

        vk::Extent3D previousExtent;
        if (gRenderResources.images->isValidHandle(m_previousResolveOutput))
            previousExtent =
                gRenderResources.images->resource(m_previousResolveOutput)
                    .extent;

        // TODO: Reset history from app when camera or scene is toggled,
        // projection changes
        bool ignoreHistory = false;
        const char *resolveDebugName = "previousResolvedIllumination";
        if (renderExtent.width != previousExtent.width ||
            renderExtent.height != previousExtent.height)
        {
            if (gRenderResources.images->isValidHandle(m_previousResolveOutput))
                gRenderResources.images->release(m_previousResolveOutput);

            // Create dummy texture to satisfy binds even though it won't be
            // read from
            m_previousResolveOutput =
                createIllumination(renderExtent, resolveDebugName);
            ignoreHistory = true;
        }
        else
            // We clear debug names each frame
            gRenderResources.images->appendDebugName(
                m_previousResolveOutput, resolveDebugName);

        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.illumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(m_previousResolveOutput)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.velocity).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(input.depth).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    gRenderResources.images->resource(ret.resolvedIllumination)
                        .view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestSampler,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.bilinearSampler,
            }},
        }};
        m_computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 5>{{
                    {input.illumination, ImageState::ComputeShaderRead},
                    {input.velocity, ImageState::ComputeShaderRead},
                    {input.depth, ImageState::ComputeShaderRead},
                    {m_previousResolveOutput, ImageState::ComputeShaderRead},
                    {ret.resolvedIllumination, ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "TemporalAntiAliasing");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{renderExtent.width, renderExtent.height, 1u});

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = m_computePass.storageSet(nextFrame);

        const uint32_t camOffset = cam.bufferOffset();

        const TaaResolveConstants constants{
            .ignoreHistory = ignoreHistory ? VK_TRUE : VK_FALSE,
            .catmullRom = m_catmullRom ? VK_TRUE : VK_FALSE,
            .colorClipping = m_colorClipping,
            .velocitySampling = m_velocitySampling,
            .luminanceWeighting = m_luminanceWeighting ? VK_TRUE : VK_FALSE,
        };

        m_computePass.record(
            cb, groupCount, descriptorSets,
            ComputePassOptionalRecordArgs{
                .dynamicOffsets = Span{&camOffset, 1},
                .specializationIndex = specializationIndex(constants),
            });

        gRenderResources.images->release(m_previousResolveOutput);
        m_previousResolveOutput = ret.resolvedIllumination;
        gRenderResources.images->preserve(m_previousResolveOutput);
    }

    return ret;
}

void TemporalAntiAliasing::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    if (gRenderResources.images->isValidHandle(m_previousResolveOutput))
        gRenderResources.images->release(m_previousResolveOutput);
}
