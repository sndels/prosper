#include "TextureDebug.hpp"

#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "utils/Hashes.hpp"
#include "utils/Profiler.hpp"
#include "utils/Ui.hpp"
#include "utils/Utils.hpp"

#include <algorithm>
#include <imgui.h>
#include <shader_structs/push_constants/texture_debug.h>

using namespace glm;
using namespace wheels;

namespace
{

const Hash<StrSpan> sStrSpanHash = Hash<StrSpan>{};

struct TextureDebugPCFlags
{
    TextureDebug::ChannelType channelType{TextureDebug::ChannelType::RGB};
    bool absBeforeRange{false};
    bool zoom{false};
    bool magnifier{false};
};

uint32_t pcFlags(TextureDebugPCFlags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.channelType;
    static_assert((uint32_t)TextureDebug::ChannelType::Count - 1 < 0b111);
    ret |= (uint32_t)flags.absBeforeRange << 3;
    ret |= (uint32_t)flags.zoom << 4;
    ret |= (uint32_t)flags.magnifier << 5;

    return ret;
}

char const *const sOutputDebugName = "TextureDebugOutput";

constexpr StaticArray<
    const char *, static_cast<size_t>(TextureDebug::ChannelType::Count)>
    sChannelTypeNames{{TEXTURE_DEBUG_CHANNEL_TYPES_STRS}};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/texture_debug.comp",
        .debugName = String{alloc, "TextureDebugCS"},
    };
}

} // namespace

TextureDebug::~TextureDebug()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    for (Buffer &b : m_readbackBuffers)
        gDevice.destroy(b);
}

void TextureDebug::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);

    for (Buffer &b : m_readbackBuffers)
    {
        b = gDevice.createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = sizeof(vec4),
                    .usage = vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .debugName = "TextureDebugReadback",
        });
        memset(b.mapped, 0, b.byteSize);
    }

    m_initialized = true;
}

void TextureDebug::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void TextureDebug::drawUi(uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    ImGui::SetNextWindowPos(ImVec2{400.f, 400.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2{50.f, 80.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("TextureDebug", nullptr);

    TargetSettings *settings = nullptr;
    {
        const Span<const String> debugNames =
            gRenderResources.images->debugNames();
        int activeNameIndex = -1;
        const Optional<StrSpan> activeName =
            gRenderResources.images->activeDebugName();
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
                gRenderResources.images->clearDebug();
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
                        gRenderResources.images->markForDebug(
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
        if (!m_targetSettings.contains(nameHash))
            m_targetSettings.insert_or_assign(nameHash, TargetSettings{});
        settings = m_targetSettings.find(nameHash);
        WHEELS_ASSERT(settings != nullptr);
    }
    WHEELS_ASSERT(settings != nullptr);

    {
        const ImageHandle activeHandle =
            gRenderResources.images->activeDebugHandle();
        int32_t maxLod = 0;
        if (gRenderResources.images->isValidHandle(activeHandle))
            maxLod = asserted_cast<int32_t>(
                         gRenderResources.images->resource(activeHandle)
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

    {
        const float *value =
            static_cast<const float *>(m_readbackBuffers[nextFrame].mapped);
        ImVec4 imVec{
            value[0],
            value[1],
            value[2],
            value[3],
        };
        ImGui::ColorButton(
            "##peekedValueButton", imVec,
            ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
        ImGui::SameLine();
        ImGui::InputFloat4("##peekedValue", &imVec.x, "%.5f");
    }

    ImGui::Checkbox("Abs before range", &settings->absBeforeRange);
    ImGui::Checkbox("Bilinear sampler", &settings->useBilinearSampler);
    ImGui::Checkbox("Zoom", &m_zoom);

    ImGui::End();
}

ImageHandle TextureDebug::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, vk::Extent2D outSize,
    wheels::Optional<glm::vec2> cursorCoord, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("TextureDebug");

    ImageHandle ret;
    {
        ret = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR8G8B8A8Unorm,
                .width = outSize.width,
                .height = outSize.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eStorage |         // TextureDebug
                    vk::ImageUsageFlagBits::eColorAttachment | // ImGui
                    vk::ImageUsageFlagBits::eTransferSrc | // Blit to swap image
                    vk::ImageUsageFlagBits::eTransferDst,  // Clear
            },
            sOutputDebugName);

        const ImageHandle inColor =
            gRenderResources.images->activeDebugHandle();

        if (!gRenderResources.images->isValidHandle(inColor) ||
            gRenderResources.images->resource(inColor).imageType !=
                vk::ImageType::e2D)
        {
            gRenderResources.images->transition(
                cb, ret, ImageState::TransferDst);

            const Image &image = gRenderResources.images->resource(ret);
            const vk::ClearColorValue clearValue{0.f, 0.f, 0.f, 1.f};
            cb.clearColorImage(
                image.handle, vk::ImageLayout::eTransferDstOptimal, &clearValue,
                1, &image.subresourceRange);
        }
        else
        {
            const Optional<StrSpan> activeName =
                gRenderResources.images->activeDebugName();

            const BufferHandle deviceReadback =
                gRenderResources.buffers->create(
                    BufferDescription{
                        .byteSize = m_readbackBuffers[nextFrame].byteSize,
                        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                    },
                    "TextureDebugReadbackDeviceBuffer");

            TargetSettings settings;
            if (activeName.has_value())
            {
                const uint64_t nameHash = sStrSpanHash(*activeName);
                const TargetSettings *settings_ptr =
                    m_targetSettings.find(nameHash);
                if (settings_ptr != nullptr)
                    settings = *settings_ptr;
            }

            const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
                scopeAlloc.child_scope(), nextFrame,
                StaticArray{{
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .imageView =
                            gRenderResources.images->resource(inColor).view,
                        .imageLayout = vk::ImageLayout::eGeneral,
                    }},
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .imageView =
                            gRenderResources.images->resource(ret).view,
                        .imageLayout = vk::ImageLayout::eGeneral,
                    }},
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .sampler = settings.useBilinearSampler
                                       ? gRenderResources.bilinearSampler
                                       : gRenderResources.nearestSampler,
                    }},
                    DescriptorInfo{vk::DescriptorImageInfo{
                        .sampler = gRenderResources.nearestSampler,
                    }},
                    DescriptorInfo{vk::DescriptorBufferInfo{
                        .buffer = gRenderResources.buffers->nativeHandle(
                            deviceReadback),
                        .range = VK_WHOLE_SIZE,
                    }},
                }});

            transition(
                WHEELS_MOV(scopeAlloc), cb,
                Transitions{
                    .images = StaticArray<ImageTransition, 2>{{
                        {inColor, ImageState::ComputeShaderRead},
                        {ret, ImageState::ComputeShaderWrite},
                    }},
                    .buffers =
                        StaticArray<BufferTransition, 1>{
                            BufferTransition{
                                deviceReadback,
                                BufferState::ComputeShaderWrite},
                        },
                });

            PROFILER_GPU_SCOPE(cb, "TextureDebug");

            const vk::Extent3D inExtent =
                gRenderResources.images->resource(inColor).extent;
            const vk::Extent3D outExtent =
                gRenderResources.images->resource(ret).extent;

            const vec2 cursorUv =
                cursorCoord.has_value()
                    ? (vec2(*cursorCoord) + 0.5f) /
                          vec2(outExtent.width, outExtent.height)
                    : vec2{};
            const TextureDebugPC pcBlock{
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
                .flags = pcFlags(TextureDebugPCFlags{
                    .channelType = settings.channelType,
                    .absBeforeRange = settings.absBeforeRange,
                    .zoom = m_zoom,
                    .magnifier = cursorCoord.has_value(),
                }),
                .cursorUv = cursorUv,
            };

            const uvec3 groupCount = m_computePass.groupCount(
                uvec3{outSize.width, outSize.height, 1u});
            m_computePass.record(cb, pcBlock, groupCount, Span{&storageSet, 1});

            gRenderResources.buffers->transition(
                cb, deviceReadback, BufferState::TransferSrc);
            // We know the host readback buffer is not used this frame so no
            // need for a barrier here

            const vk::BufferCopy region{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = m_readbackBuffers[nextFrame].byteSize,
            };
            cb.copyBuffer(
                gRenderResources.buffers->nativeHandle(deviceReadback),
                m_readbackBuffers[nextFrame].handle, 1, &region);

            gRenderResources.buffers->release(deviceReadback);
        }
    }

    return ret;
}

bool TextureDebug::textureSelected()
{
    return gRenderResources.images->activeDebugName().has_value();
}
