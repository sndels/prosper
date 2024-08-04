#include "TemporalAntiAliasing.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Ui.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"
#include "Utils.hpp"

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

enum BindingSet : uint32_t
{
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t flags{0};

    struct Flags
    {
        bool ignoreHistory{false};
        bool catmullRom{false};
        TemporalAntiAliasing::ColorClippingType colorClipping{
            TemporalAntiAliasing::ColorClippingType::None};
        TemporalAntiAliasing::VelocitySamplingType velocitySampling{
            TemporalAntiAliasing::VelocitySamplingType::Center};
        bool luminanceWeighting{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.ignoreHistory;
    ret |= (uint32_t)flags.catmullRom << 1;
    ret |= (uint32_t)flags.colorClipping << 2;
    static_assert(
        (uint32_t)TemporalAntiAliasing::ColorClippingType::Count - 1 < 0b11);
    ret |= (uint32_t)flags.velocitySampling << 4;
    static_assert(
        (uint32_t)TemporalAntiAliasing::VelocitySamplingType::Count - 1 < 0b11);
    ret |= (uint32_t)flags.luminanceWeighting << 6;

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

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
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

        m_computePass.record(
            cb,
            PCBlock{
                .flags = pcFlags(PCBlock::Flags{
                    .ignoreHistory = ignoreHistory,
                    .catmullRom = m_catmullRom,
                    .colorClipping = m_colorClipping,
                    .velocitySampling = m_velocitySampling,
                    .luminanceWeighting = m_luminanceWeighting,
                }),
            },
            groupCount, descriptorSets, Span{&camOffset, 1});

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
