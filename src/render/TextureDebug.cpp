#include "TextureDebug.hpp"

#include <imgui.h>

#include <algorithm>
#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../utils/Hashes.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Ui.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const Hash<StrSpan> sStrSpanHash = Hash<StrSpan>{};

struct PCBlock
{
    ivec2 inRes{};
    ivec2 outRes{};

    vec2 range{0.f, 1.f};
    uint32_t lod{0};
    uint32_t flags{0};

    struct Flags
    {
        TextureDebug::ChannelType channelType{TextureDebug::ChannelType::RGB};
        bool absBeforeRange{false};
        bool zoom{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.channelType;
    // Three bits reserved for the channel type
    static_assert((uint32_t)TextureDebug::ChannelType::Count - 1 < 0b111);
    ret |= (uint32_t)flags.absBeforeRange << 3;
    ret |= (uint32_t)flags.zoom << 4;

    return ret;
}

char const *const sOutputDebugName = "TextureDebugOutput";

constexpr std::array<
    const char *, static_cast<size_t>(TextureDebug::ChannelType::Count)>
    sChannelTypeNames = {TEXTURE_DEBUG_CHANNEL_TYPES_STRS};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/texture_debug.comp",
        .debugName = String{alloc, "TextureDebugCS"},
    };
}

} // namespace

TextureDebug::TextureDebug(
    Allocator &alloc, ScopedScratch scopeAlloc, Device *device,
    RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc)
: _resources{resources}
, _computePass{WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc, shaderDefinitionCallback}
, _targetSettings{alloc}
{
    WHEELS_ASSERT(_resources != nullptr);
}

void TextureDebug::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void TextureDebug::drawUi()
{
    ImGui::SetNextWindowPos(ImVec2{400.f, 400.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2{50.f, 80.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("TextureDebug", nullptr);

    TargetSettings *settings = nullptr;
    {
        const Span<const String> debugNames = _resources->images.debugNames();
        int activeNameIndex = -1;
        const Optional<StrSpan> activeName =
            _resources->images.activeDebugName();
        if (activeName.has_value())
        {
            int i = 0;
            for (const String &name : debugNames)
            {
                if (*activeName == name)
                {
                    activeNameIndex = i;
                    break;
                }
                ++i;
            }
        }

        const char *emptySlotName = "##TextureDebugDropDownEmptyName";
        const char *comboTitle =
            activeNameIndex >= 0 ? debugNames.data()[activeNameIndex].c_str()
                                 : emptySlotName;
        if (ImGui::BeginCombo("##TextureDebugDropDown", comboTitle, 0))
        {
            bool selected = activeNameIndex == -1;
            if (ImGui::Selectable(emptySlotName, selected))
            {
                activeNameIndex = -1;
                _resources->images.clearDebug();
            }
            if (selected)
                ImGui::SetItemDefaultFocus();

            int i = 0;
            for (const String &name : debugNames)
            {
                selected = activeNameIndex == i;
                if (name != sOutputDebugName)
                {
                    if (ImGui::Selectable(name.c_str(), selected))
                    {
                        activeNameIndex = i;
                        _resources->images.markForDebug(
                            debugNames.data()[activeNameIndex]);
                    }

                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ++i;
            }
            ImGui::EndCombo();
        }

        const uint64_t nameHash = sStrSpanHash(StrSpan{comboTitle});
        if (!_targetSettings.contains(nameHash))
            _targetSettings.insert_or_assign(nameHash, TargetSettings{});
        settings = _targetSettings.find(nameHash);
        WHEELS_ASSERT(settings != nullptr);
    }
    WHEELS_ASSERT(settings != nullptr);

    {
        const ImageHandle activeHandle = _resources->images.activeDebugHandle();
        int32_t maxLod = 0;
        if (_resources->images.isValidHandle(activeHandle))
            maxLod =
                asserted_cast<int32_t>(_resources->images.resource(activeHandle)
                                           .subresourceRange.levelCount) -
                1;
        ImGui::DragInt("LoD##TextureDebug", &settings->lod, 0.02f, 0, maxLod);
        settings->lod = std::clamp(settings->lod, 0, maxLod);
    }

    enumDropdown("Channel", settings->channelType, sChannelTypeNames);

    {
        // Having drag speed react to the absolute range makes this nicer to use
        // Zero makes things misbehave so avoid it
        const float rangeLen =
            std::max(std::abs(settings->range[1] - settings->range[0]), 1e-3f);
        const float rangeSpeed = rangeLen * 1e-3f;
        // Adapt formatting to range, this also controls actual precicion of the
        // values we get
        const char *format = rangeLen < 0.01 ? "%.6f" : "%.3f";
        ImGui::DragFloat2(
            "Range##TextureDebug", &settings->range[0], rangeSpeed, -1e6f, 1e6f,
            format);
        // Don't allow the limits swapping places
        settings->range[0] = std::min(settings->range[0], settings->range[1]);
        settings->range[1] = std::max(settings->range[0], settings->range[1]);
    }

    ImGui::Checkbox("Abs before range", &settings->absBeforeRange);
    ImGui::Checkbox("Bilinear sampler", &settings->useBilinearSampler);
    ImGui::Checkbox("Zoom", &_zoom);

    ImGui::End();
}

ImageHandle TextureDebug::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, vk::Extent2D outSize,
    uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    ImageHandle ret;
    {
        ret = createOutput(outSize);
        const ImageHandle inColor = _resources->images.activeDebugHandle();

        if (!_resources->images.isValidHandle(inColor) ||
            _resources->images.resource(inColor).imageType !=
                vk::ImageType::e2D)
        {
            _resources->images.transition(cb, ret, ImageState::TransferDst);

            const Image &image = _resources->images.resource(ret);
            const vk::ClearColorValue clearValue{0.f, 0.f, 0.f, 1.f};
            cb.clearColorImage(
                image.handle, vk::ImageLayout::eTransferDstOptimal, &clearValue,
                1, &image.subresourceRange);
        }
        else
        {
            const Optional<StrSpan> activeName =
                _resources->images.activeDebugName();

            TargetSettings settings;
            if (activeName.has_value())
            {
                const uint64_t nameHash = sStrSpanHash(*activeName);
                const TargetSettings *settings_ptr =
                    _targetSettings.find(nameHash);
                if (settings_ptr != nullptr)
                    settings = *settings_ptr;
            }

            _computePass.updateDescriptorSet(
                WHEELS_MOV(scopeAlloc), nextFrame,
                StaticArray{
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .imageView = _resources->images.resource(inColor).view,
                        .imageLayout = vk::ImageLayout::eGeneral,
                    }},
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .imageView = _resources->images.resource(ret).view,
                        .imageLayout = vk::ImageLayout::eGeneral,
                    }},
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .sampler = settings.useBilinearSampler
                                       ? _resources->bilinearSampler
                                       : _resources->nearestSampler,
                    }},
                });

            transition<2>(
                *_resources, cb,
                {
                    {inColor, ImageState::ComputeShaderRead},
                    {ret, ImageState::ComputeShaderWrite},
                });

            const auto _s = profiler->createCpuGpuScope(cb, "TextureDebug");

            const vk::Extent3D inExtent =
                _resources->images.resource(inColor).extent;
            const vk::Extent3D outExtent =
                _resources->images.resource(ret).extent;

            const PCBlock pcBlock{
                .inRes =
                    uvec2{
                        asserted_cast<int32_t>(inExtent.width),
                        asserted_cast<int32_t>(inExtent.height),
                    },
                .outRes =
                    uvec2{
                        asserted_cast<int32_t>(outExtent.width),
                        asserted_cast<int32_t>(outExtent.height),
                    },
                .range = settings.range,
                .lod = asserted_cast<uint32_t>(settings.lod),
                .flags = pcFlags(PCBlock::Flags{
                    .channelType = settings.channelType,
                    .absBeforeRange = settings.absBeforeRange,
                    .zoom = _zoom,
                })};

            const uvec3 extent = uvec3{outSize.width, outSize.height, 1u};
            const vk::DescriptorSet storageSet =
                _computePass.storageSet(nextFrame);
            _computePass.record(cb, pcBlock, extent, Span{&storageSet, 1});
        }
    }

    return ret;
}

ImageHandle TextureDebug::createOutput(vk::Extent2D size)
{
    return _resources->images.create(
        ImageDescription{
            .format = vk::Format::eR8G8B8A8Unorm,
            .width = size.width,
            .height = size.height,
            .usageFlags =
                vk::ImageUsageFlagBits::eStorage |         // TextureDebug
                vk::ImageUsageFlagBits::eColorAttachment | // ImGui
                vk::ImageUsageFlagBits::eTransferSrc |     // Blit to swap image
                vk::ImageUsageFlagBits::eTransferDst,      // Clear
        },
        sOutputDebugName);
}
