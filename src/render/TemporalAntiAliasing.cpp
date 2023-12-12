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

constexpr std::array<
    const char *,
    static_cast<size_t>(TemporalAntiAliasing::ColorClippingType::Count)>
    sColorClippingTypeNames = {"None", COLOR_CLIPPING_TYPE_STRS};

enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t colorClipping{0};
    uint32_t flags{0};

    struct Flags
    {
        bool ignoreHistory{false};
        bool catmullRom{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.ignoreHistory;
    ret |= (uint32_t)flags.catmullRom << 1;

    return ret;
}

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

    const size_t len = 140;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendEnumVariantsAsDefines(
        defines, "ColorClipping",
        Span{sColorClippingTypeNames.data(), sColorClippingTypeNames.size()});
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/taa_resolve.comp",
        .debugName = String{alloc, "TaaResolveCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

TemporalAntiAliasing::TemporalAntiAliasing(
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

void TemporalAntiAliasing::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        Span{&camDSLayout, 1});
}

void TemporalAntiAliasing::drawUi()
{
    uint32_t *currentType = reinterpret_cast<uint32_t *>(&_colorClipping);
    if (ImGui::BeginCombo(
            "Color clipping", sColorClippingTypeNames[*currentType]))
    {
        for (auto i = 0u; i < static_cast<uint32_t>(ColorClippingType::Count);
             ++i)
        {
            bool selected = *currentType == i;
            if (ImGui::Selectable(sColorClippingTypeNames[i], &selected))
                _colorClipping = static_cast<ColorClippingType>(i);
        }
        ImGui::EndCombo();
    }
    ImGui::Checkbox("Catmull-Rom history samples", &_catmullRom);
}

TemporalAntiAliasing::Output TemporalAntiAliasing::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Camera &cam,
    const Input &input, const uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.illumination);

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
                .imageView =
                    _resources->images.resource(input.illumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    _resources->images.resource(_previousResolveOutput).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = _resources->images.resource(input.velocity).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView =
                    _resources->images.resource(ret.resolvedIllumination).view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _resources->nearestSampler,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _resources->bilinearSampler,
            }},
        };
        _computePass.updateDescriptorSet(
            WHEELS_MOV(scopeAlloc), nextFrame, descriptorInfos);

        transition<4>(
            *_resources, cb,
            {
                {input.illumination, ImageState::ComputeShaderRead},
                {input.velocity, ImageState::ComputeShaderRead},
                {_previousResolveOutput, ImageState::ComputeShaderRead},
                {ret.resolvedIllumination, ImageState::ComputeShaderWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "TemporalAntiAliasing");

        const uvec3 groups = uvec3{
            (uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u + 1u,
            1u};

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const uint32_t camOffset = cam.bufferOffset();

        _computePass.record(
            cb,
            PCBlock{
                .colorClipping = static_cast<uint32_t>(_colorClipping),
                .flags = pcFlags(PCBlock::Flags{
                    .ignoreHistory = ignoreHistory,
                    .catmullRom = _catmullRom,
                }),
            },
            groups, descriptorSets, Span{&camOffset, 1});

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
