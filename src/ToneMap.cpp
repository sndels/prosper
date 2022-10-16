#include "ToneMap.hpp"

#include <glm/glm.hpp>

#include <fstream>

#include "Utils.hpp"

using namespace glm;

ToneMap::ToneMap(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating ToneMap\n");

    if (!compileShaders())
        throw std::runtime_error("ToneMap shader compilation failed");

    const std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    recreateSwapchainRelated(swapConfig);
}

ToneMap::~ToneMap()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void ToneMap::recompileShaders()
{
    if (compileShaders())
    {
        destroyPipelines();
        createPipelines();
    }
}

bool ToneMap::compileShaders()
{
    fprintf(stderr, "Compiling ToneMap shaders\n");

    const auto compSM =
        _device->compileShaderModule(Device::CompileShaderModuleArgs{
            .relPath = "shader/tone_map.comp",
            .debugName = "tonemapCS",
        });

    if (compSM)
    {
        _device->logical().destroy(_compSM);

        _compSM = *compSM;

        return true;
    }

    return false;
}

void ToneMap::recreateSwapchainRelated(const SwapchainConfig &swapConfig)
{
    destroySwapchainRelated();
    createOutputImage(swapConfig);
    createDescriptorSet(swapConfig);
    createPipelines();
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer ToneMap::execute(
    const uint32_t nextImage, Profiler *profiler) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    {
        const auto _s = profiler->createScope(buffer, "ToneMap");

        const std::array<vk::ImageMemoryBarrier2, 2> barriers{
            _resources->images.sceneColor.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
            _resources->images.toneMapped.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
        };

        buffer.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        buffer.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, 1,
            &_descriptorSets[nextImage], 0, nullptr);

        const auto &extent = _resources->images.sceneColor.extent;
        const auto groups =
            (glm::uvec2{extent.width, extent.height} - 1u) / 16u + 1u;
        buffer.dispatch(groups.x, groups.y, 1);
    }

    buffer.end();

    return buffer;
}

void ToneMap::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        if (!_commandBuffers.empty())
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                asserted_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        destroyPipelines();

        // Descriptor sets are cleaned up when the pool is destroyed
        _device->destroy(_resources->images.toneMapped);
    }
}

void ToneMap::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void ToneMap::createOutputImage(const SwapchainConfig &swapConfig)
{
    _resources->images.toneMapped = _device->createImage(ImageCreateInfo{
        .format = vk::Format::eR8G8B8A8Unorm,
        .width = swapConfig.extent.width,
        .height = swapConfig.extent.height,
        .usageFlags =
            vk::ImageUsageFlagBits::eStorage |         // ToneMap
            vk::ImageUsageFlagBits::eColorAttachment | // ImGui
            vk::ImageUsageFlagBits::eTransferSrc,      // Blit to swap image
        .debugName = "toneMapped",
    });
}

void ToneMap::createDescriptorSet(const SwapchainConfig &swapConfig)
{
    const std::vector<vk::DescriptorSetLayout> layouts(
        swapConfig.imageCount, _descriptorSetLayout);
    _descriptorSets =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _resources->descriptorPools.swapchainRelated,
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        });

    const vk::DescriptorImageInfo colorInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo mappedInfo{
        .imageView = _resources->images.toneMapped.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    std::vector<vk::WriteDescriptorSet> descriptorWrites;
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
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &_descriptorSetLayout,
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

void ToneMap::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount,
        });
}
