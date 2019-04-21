#include "VkUtils.hpp"

namespace {
    std::pair<vk::AccessFlags, vk::AccessFlags> accessMasks(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout)
    {
        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            return std::pair{
                vk::AccessFlags{},
                vk::AccessFlagBits::eTransferWrite
            };
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
                   newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            return std::pair{
                vk::AccessFlags{},
                vk::AccessFlagBits::eDepthStencilAttachmentRead |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eTransferSrcOptimal) {
            return std::pair{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eTransferRead
            };
        } else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair{
                vk::AccessFlagBits::eTransferRead,
                vk::AccessFlagBits::eShaderRead
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::ePresentSrcKHR) {
            return std::pair{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eMemoryRead
            };
        } else
            throw std::runtime_error("Unsupported layout transition");
    }

    std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags> stageMasks(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout)
    {
        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            return std::pair{
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer
            };
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
                   newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            return std::pair{
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eEarlyFragmentTests
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eTransferSrcOptimal) {
            return std::pair{
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer
            };
        } else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair{
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            return std::pair{
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader
            };
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::ePresentSrcKHR) {
            return std::pair{
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer
            };
        } else
            throw std::runtime_error("Unsupported layout transition");
    }
}

void transitionImageLayout(const vk::Image& image, const vk::ImageSubresourceRange& subresourceRange, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, const vk::CommandBuffer& commandBuffer)
{
    const auto [srcAccessMask, dstAccessMask] = accessMasks(oldLayout, newLayout);
    const auto [srcStageMask, dstStageMask] = stageMasks(oldLayout, newLayout);

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
        {}, // dependencyFlags
        0, nullptr, // memoryBarriers
        0, nullptr, // bufferMemoryBarriers
        1, &barrier
    );
}
