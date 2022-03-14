#include "Resources.hpp"

vk::BufferMemoryBarrier2 TexelBuffer::transitionBarrier(
    const BufferState &newState)
{
    const vk::BufferMemoryBarrier2 barrier{
        .srcStageMask = state.stageMask,
        .srcAccessMask = state.accessMask,
        .dstStageMask = newState.stageMask,
        .dstAccessMask = newState.accessMask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = handle,
        .offset = 0,
        .size = size,
    };

    state = newState;

    return barrier;
}

void TexelBuffer::transition(
    const vk::CommandBuffer buffer, const BufferState &newState)
{
    auto barrier = transitionBarrier(newState);
    buffer.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });
}

vk::ImageMemoryBarrier2 Image::transitionBarrier(const ImageState &newState)
{
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = state.stageMask,
        .srcAccessMask = state.accessMask,
        .dstStageMask = newState.stageMask,
        .dstAccessMask = newState.accessMask,
        .oldLayout = state.layout,
        .newLayout = newState.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresourceRange = subresourceRange,
    };

    state = newState;

    return barrier;
}

void Image::transition(
    const vk::CommandBuffer buffer, const ImageState &newState)
{
    auto barrier = transitionBarrier(newState);
    buffer.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}
