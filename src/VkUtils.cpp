#include "VkUtils.hpp"

void transitionImageLayout(const vk::CommandBuffer &commandBuffer,
                           const vk::Image &image,
                           const vk::ImageSubresourceRange &subresourceRange,
                           const vk::ImageLayout oldLayout,
                           const vk::ImageLayout newLayout,
                           const vk::AccessFlags srcAccessMask,
                           const vk::AccessFlags dstAccessMask,
                           const vk::PipelineStageFlags srcStageMask,
                           const vk::PipelineStageFlags dstStageMask) {
    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange};
    commandBuffer.pipelineBarrier(srcStageMask, dstStageMask,
                                  vk::DependencyFlags{}, 0, nullptr, 0, nullptr,
                                  1, &barrier);
}
