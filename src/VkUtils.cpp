#include "VkUtils.hpp"

void transitionImageLayout(
    const vk::CommandBuffer &commandBuffer, const vk::Image &image,
    const vk::ImageSubresourceRange &subresourceRange,
    const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
    const vk::AccessFlags2KHR srcAccessMask,
    const vk::AccessFlags2KHR dstAccessMask,
    const vk::PipelineStageFlags2KHR srcStageMask,
    const vk::PipelineStageFlags2KHR dstStageMask)
{
    const vk::ImageMemoryBarrier2KHR barrier{
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange};
    commandBuffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

void checkSuccess(vk::Result result, const char *source)
{
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            std::string(source) + " failed: " + vk::to_string(result) + "!");
    }
}
