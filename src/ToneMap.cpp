#include "ToneMap.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

struct PCBlock
{
    float exposure{1.f};
};

} // namespace

ToneMap::ToneMap(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const vk::Extent2D &renderExtent)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating ToneMap\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("ToneMap shader compilation failed");

    createDescriptorSets();

    recreate(renderExtent);
}

ToneMap::~ToneMap()
{
    if (_device != nullptr)
    {
        destroyViewportRelated();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void ToneMap::recompileShaders(ScopedScratch scopeAlloc)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipelines();
        createPipelines();
    }
}

bool ToneMap::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling ToneMap shaders\n");

    const auto compSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/tone_map.comp",
                                      .debugName = "tonemapCS",
                                  });

    if (compSM.has_value())
    {
        _device->logical().destroy(_compSM);

        _compSM = *compSM;

        return true;
    }

    return false;
}

void ToneMap::recreate(const vk::Extent2D &renderExtent)
{
    destroyViewportRelated();
    createOutputImage(renderExtent);
    updateDescriptorSets();
    createPipelines();
}

void ToneMap::drawUi()
{
    ImGui::DragFloat("Exposure", &_exposure, 0.5f, 0.001f, 10000.f);
}

void ToneMap::record(
    vk::CommandBuffer cb, const uint32_t nextFrame, Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const auto _s = profiler->createCpuGpuScope(cb, "ToneMap");

        const StaticArray barriers{
            _resources->staticImages.sceneColor.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
            _resources->staticImages.toneMapped.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
        };

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, 1,
            &_descriptorSets[nextFrame], 0, nullptr);

        const PCBlock pcBlock{
            .exposure = _exposure,
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
            sizeof(PCBlock), &pcBlock);

        const auto &extent = _resources->staticImages.sceneColor.extent;
        const auto groups =
            (glm::uvec2{extent.width, extent.height} - 1u) / 16u + 1u;
        cb.dispatch(groups.x, groups.y, 1);
    }
}

void ToneMap::destroyViewportRelated()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        // Descriptor sets are cleaned up when the pool is destroyed
        _device->destroy(_resources->staticImages.toneMapped);
    }
}

void ToneMap::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void ToneMap::createOutputImage(const vk::Extent2D &renderExtent)
{
    _resources->staticImages.toneMapped = _device->createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR8G8B8A8Unorm,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eStorage |         // ToneMap
                    vk::ImageUsageFlagBits::eColorAttachment | // ImGui
                    vk::ImageUsageFlagBits::eTransferSrc, // Blit to swap image
            },
        .debugName = "toneMapped",
    });
}

void ToneMap::createDescriptorSets()
{
    const StaticArray layoutBindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    _resources->staticDescriptorsAlloc.allocate(layouts, _descriptorSets);
}

void ToneMap::updateDescriptorSets()
{
    const vk::DescriptorImageInfo colorInfo{
        .imageView = _resources->staticImages.sceneColor.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo mappedInfo{
        .imageView = _resources->staticImages.toneMapped.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    StaticArray<vk::WriteDescriptorSet, MAX_FRAMES_IN_FLIGHT * 2>
        descriptorWrites;
    for (const auto &ds : _descriptorSets)
    {
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &colorInfo,
        });
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &mappedInfo,
        });
    }
    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void ToneMap::createPipelines()
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

    {
        auto pipeline = _device->logical().createComputePipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "ToneMap",
            });
    }
}
