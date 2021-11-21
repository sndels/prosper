#include "ToneMap.hpp"

#include <glm/glm.hpp>

#include <fstream>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

ToneMap::ToneMap(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig)
: _device{device}
, _resources{resources}
{
    const std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{
        {{.binding = 0,
          .descriptorType = vk::DescriptorType::eStorageImage,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute},
         {.binding = 1,
          .descriptorType = vk::DescriptorType::eStorageImage,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute}}};
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data()});

    recreateSwapchainRelated(swapConfig);
}

ToneMap::~ToneMap()
{
    if (_device)
    {
        destroySwapchainRelated();
        _device->logical().destroy(_descriptorSetLayout);
    }
}

void ToneMap::recreateSwapchainRelated(const SwapchainConfig &swapConfig)
{
    destroySwapchainRelated();
    createOutputImage(swapConfig);
    createDescriptorSet(swapConfig);
    createPipelines(swapConfig);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer ToneMap::execute(const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    buffer.beginDebugUtilsLabelEXT(
        vk::DebugUtilsLabelEXT{.pLabelName = "ToneMap"});

    transitionImageLayout(
        buffer, _resources->images.sceneColor.handle,
        vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1},
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits2KHR::eColorAttachmentWrite,
        vk::AccessFlagBits2KHR::eShaderRead,
        vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2KHR::eComputeShader);

    transitionImageLayout(
        buffer, _resources->images.toneMapped.handle,
        vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1},
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits2KHR::eMemoryRead,
        vk::AccessFlagBits2KHR::eShaderWrite,
        vk::PipelineStageFlagBits2KHR::eTopOfPipe,
        vk::PipelineStageFlagBits2KHR::eComputeShader);

    buffer.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, 1,
        &_descriptorSets[nextImage], 0, nullptr);

    const auto &extent = _resources->images.sceneColor.extent;
    const auto groups = glm::uvec2{extent.width, extent.height} / 16u;
    buffer.dispatch(groups.x, groups.y, 1);

    buffer.endDebugUtilsLabelEXT(); // ToneMap

    buffer.end();

    return buffer;
}

void ToneMap::destroySwapchainRelated()
{
    if (_device)
    {
        if (_commandBuffers.size() > 0)
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                static_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        _device->logical().destroy(_pipeline);
        _device->logical().destroy(_pipelineLayout);
        // Descriptor sets are cleaned up when the pool is destroyed
        _device->destroy(_resources->images.toneMapped);
    }
}

void ToneMap::createOutputImage(const SwapchainConfig &swapConfig)
{
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1};

    _resources->images.toneMapped = _device->createImage(
        "toneMapped", swapConfig.extent, vk::Format::eR8G8B8A8Unorm,
        subresourceRange, vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
        vk::ImageCreateFlagBits{},
        vk::ImageUsageFlagBits::eStorage |             // ToneMap
            vk::ImageUsageFlagBits::eColorAttachment | // ImGui
            vk::ImageUsageFlagBits::eTransferSrc,      // Blit to swap image
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    auto buffer = _device->beginGraphicsCommands();
    // Before barrier expects this state
    transitionImageLayout(
        buffer, _resources->images.toneMapped.handle,
        vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1},
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2KHR::eNone, vk::AccessFlagBits2KHR::eMemoryRead,
        vk::PipelineStageFlagBits2KHR::eTopOfPipe,
        vk::PipelineStageFlagBits2KHR::eTopOfPipe);
    _device->endGraphicsCommands(buffer);
}

void ToneMap::createDescriptorSet(const SwapchainConfig &swapConfig)
{
    const std::vector<vk::DescriptorSetLayout> layouts(
        swapConfig.imageCount, _descriptorSetLayout);
    _descriptorSets =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _resources->descriptorPools.swapchainRelated,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()});

    std::vector<vk::WriteDescriptorSet> descriptorWrites;
    for (size_t i = 0; i < _descriptorSets.size(); ++i)
    {
        {
            vk::DescriptorImageInfo info{
                .imageView = _resources->images.sceneColor.view,
                .imageLayout = vk::ImageLayout::eGeneral};
            descriptorWrites.push_back(
                {.dstSet = _descriptorSets[i],
                 .dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &info});
        }

        {
            vk::DescriptorImageInfo info{
                .imageView = _resources->images.toneMapped.view,
                .imageLayout = vk::ImageLayout::eGeneral};
            descriptorWrites.push_back(
                {.dstSet = _descriptorSets[i],
                 .dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &info});
        }
    }
    _device->logical().updateDescriptorSets(
        static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
        0, nullptr);
}

void ToneMap::createPipelines(const SwapchainConfig &swapConfig)
{
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &_descriptorSetLayout});

    const auto compSPV = readFileBytes(binPath("shader/tone_map.comp.spv"));
    const vk::ShaderModule compSM =
        createShaderModule(_device->logical(), compSPV);

    const vk::ComputePipelineCreateInfo createInfo{
        .stage =
            {.stage = vk::ShaderStageFlagBits::eCompute,
             .module = compSM,
             .pName = "main"},
        .layout = _pipelineLayout};

    {
        auto pipeline = _device->logical().createComputePipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;
    }

    _device->logical().destroy(compSM);
}

void ToneMap::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}