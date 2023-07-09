#include "TextureDebug.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <algorithm>
#include <fstream>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    ivec2 inRes{};
    ivec2 outRes{};

    vec2 range{0.f, 1.f};
    uint32_t lod{0};
    uint32_t channelType{0};

    uint32_t absBeforeRange{0};
};

char const *sOutputDebugName = "TextureDebugOutput";

constexpr std::array<
    const char *, static_cast<size_t>(TextureDebug::ChannelType::Count)>
    sChannelTypeNames = {TEXTURE_DEBUG_CHANNEL_TYPES_STRS};

} // namespace

TextureDebug::TextureDebug(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating TextureDebug\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("TextureDebug shader compilation failed");

    vk::SamplerCreateInfo info{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    _linearSampler = _device->logical().createSampler(info);

    info.magFilter = vk::Filter::eNearest;
    info.minFilter = vk::Filter::eNearest;
    _nearestSampler = _device->logical().createSampler(info);

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipelines();
}

TextureDebug::~TextureDebug()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);
        _device->logical().destroy(_nearestSampler);
        _device->logical().destroy(_linearSampler);
        _device->logical().destroy(_compSM);
    }
}

void TextureDebug::recompileShaders(ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipelines();
    }
}

bool TextureDebug::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling TextureDebug shaders\n");

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/texture_debug.comp",
                .debugName = "TextureDebugCS",
            });

    if (compResult.has_value())
    {
        _device->logical().destroy(_compSM);

        ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(PCBlock) == reflection.pushConstantsBytesize());

        _compSM = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}

void TextureDebug::drawUi()
{
    ImGui::SetNextWindowPos(ImVec2{400.f, 400.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2{50.f, 80.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("TextureDebug", nullptr);

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
    }

    {
        const Optional<ImageHandle> activeHandle =
            _resources->images.activeDebugHandle();
        int32_t maxLod = 0;
        if (activeHandle.has_value())
            maxLod = asserted_cast<int32_t>(
                         _resources->images.resource(*activeHandle)
                             .subresourceRange.levelCount) -
                     1;
        ImGui::DragInt("LoD##TextureDebug", &_lod, 0.02f, 0, maxLod);
        _lod = std::clamp(_lod, 0, maxLod);
    }

    {
        auto *currentType = reinterpret_cast<uint32_t *>(&_channelType);
        if (ImGui::BeginCombo(
                "Channel##TextureDebug", sChannelTypeNames[*currentType]))
        {
            for (auto i = 0u; i < static_cast<uint32_t>(ChannelType::Count);
                 ++i)
            {
                bool selected = *currentType == i;
                if (ImGui::Selectable(sChannelTypeNames[i], &selected))
                    _channelType = static_cast<ChannelType>(i);
            }
            ImGui::EndCombo();
        }
    }

    {
        // Having drag speed react to the absolute range makes this nicer to use
        // Zero makes things misbehave so avoid it
        const float rangeLen = std::max(std::abs(_range[1] - _range[0]), 1e-3f);
        const float rangeSpeed = rangeLen * 1e-3f;
        // Adapt formatting to range, this also controls actual precicion of the
        // values we get
        const char *format = rangeLen < 0.01 ? "%.6f" : "%.3f";
        ImGui::DragFloat2(
            "Range##TextureDebug", &_range[0], rangeSpeed, -1e6f, 1e6f, format);
        // Don't allow the limits swapping places
        _range[0] = std::min(_range[0], _range[1]);
        _range[1] = std::max(_range[0], _range[1]);
    }

    ImGui::Checkbox("Abs before range", &_absBeforeRange);
    ImGui::Checkbox("Linear sampler", &_useLinearSampler);

    ImGui::End();
}

ImageHandle TextureDebug::record(
    vk::CommandBuffer cb, vk::Extent2D outSize, uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    ImageHandle ret;
    {
        ret = createOutput(outSize);
        const Optional<ImageHandle> inColor =
            _resources->images.activeDebugHandle();

        if (!inColor.has_value() ||
            _resources->images.resource(*inColor).imageType !=
                vk::ImageType::e2D)
        {
            _resources->images.transition(
                cb, ret,
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eCopy,
                    .accessMask = vk::AccessFlagBits2::eTransferWrite,
                    .layout = vk::ImageLayout::eTransferDstOptimal,
                });

            const Image &image = _resources->images.resource(ret);
            const vk::ClearColorValue clearValue{0.f, 0.f, 0.f, 1.f};
            cb.clearColorImage(
                image.handle, vk::ImageLayout::eTransferDstOptimal, &clearValue,
                1, &image.subresourceRange);
        }
        else
        {
            const BoundImages images{
                .inColor = *inColor,
                .outColor = ret,
            };
            updateDescriptorSet(nextFrame, images);

            recordBarriers(cb, images);

            const auto _s = profiler->createCpuGpuScope(cb, "TextureDebug");

            cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

            cb.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, 1,
                &_descriptorSets[nextFrame], 0, nullptr);

            const vk::Extent3D inExtent =
                _resources->images.resource(images.inColor).extent;
            const vk::Extent3D outExtent =
                _resources->images.resource(images.outColor).extent;

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
                .range = _range,
                .lod = asserted_cast<uint32_t>(_lod),
                .channelType = static_cast<uint32_t>(_channelType),
                .absBeforeRange = static_cast<uint32_t>(_absBeforeRange),
            };
            cb.pushConstants(
                _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                sizeof(PCBlock), &pcBlock);

            const auto groups =
                (glm::uvec2{outSize.width, outSize.height} - 1u) / 16u + 1u;
            cb.dispatch(groups.x, groups.y, 1);
        }
    }

    return ret;
}

void TextureDebug::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void TextureDebug::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflection->generateLayoutBindings(
            scopeAlloc, 0, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void TextureDebug::updateDescriptorSet(
    uint32_t nextFrame, const BoundImages &images)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorImageInfo colorInfo{
        .imageView = _resources->images.resource(images.inColor).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo mappedInfo{
        .imageView = _resources->images.resource(images.outColor).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo samplerInfo{
        .sampler = _useLinearSampler ? _linearSampler : _nearestSampler,
    };

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites<3>(
            0, _descriptorSets[nextFrame],
            {
                Pair{0u, DescriptorInfo{colorInfo}},
                Pair{1u, DescriptorInfo{mappedInfo}},
                Pair{2u, DescriptorInfo{samplerInfo}},
            });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void TextureDebug::createPipelines()
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &_descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::ComputePipelineCreateInfo createInfo{
        .stage =
            {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = _compSM,
                .pName = "main",
            },
        .layout = _pipelineLayout,
    };

    _pipeline =
        createComputePipeline(_device->logical(), createInfo, "TextureDebug");
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
                vk::ImageUsageFlagBits::eTransferSrc | //  Blit to swap image
                vk::ImageUsageFlagBits::eTransferDst,  //  Clear
        },
        sOutputDebugName);
}

void TextureDebug::recordBarriers(
    vk::CommandBuffer cb, const BoundImages &images) const
{
    const StaticArray barriers{
        _resources->images.transitionBarrier(
            images.inColor,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderSampledRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            images.outColor,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });
}
