#include "VkUtils.hpp"

void transitionImageLayout(const vk::CommandBuffer& commandBuffer, const vk::Image& image, const vk::ImageSubresourceRange& subresourceRange, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, const vk::AccessFlags srcAccessMask, const vk::AccessFlags dstAccessMask, const vk::PipelineStageFlags srcStageMask, const vk::PipelineStageFlags dstStageMask)
{
    const vk::ImageMemoryBarrier barrier{
        srcAccessMask,
        dstAccessMask,
        oldLayout,
        newLayout,
        VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
        image,
        subresourceRange
    };
    commandBuffer.pipelineBarrier(
        srcStageMask,
        dstStageMask,
        vk::DependencyFlags{},
        0, nullptr, // memoryBarriers
        0, nullptr, // bufferMemoryBarriers
        1, &barrier
    );
}
