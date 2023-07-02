#include "Resources.hpp"

namespace
{

vk::BufferMemoryBarrier2 bufferTransitionBarrier(
    vk::Buffer &buffer, BufferState &oldState, vk::DeviceSize size,
    const BufferState &newState)
{
    const vk::BufferMemoryBarrier2 barrier{
        .srcStageMask = oldState.stageMask,
        .srcAccessMask = oldState.accessMask,
        .dstStageMask = newState.stageMask,
        .dstAccessMask = newState.accessMask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = 0,
        .size = size,
    };

    oldState = newState;

    return barrier;
}

void bufferTransition(
    const vk::CommandBuffer cb, vk::Buffer &buffer, BufferState &oldState,
    vk::DeviceSize size, const BufferState &newState)

{
    const vk::BufferMemoryBarrier2 barrier =
        bufferTransitionBarrier(buffer, oldState, size, newState);
    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });
}

} // namespace

vk::BufferMemoryBarrier2 Buffer::transitionBarrier(const BufferState &newState)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->byteSize, newState);
}

void Buffer::transition(const vk::CommandBuffer cb, const BufferState &newState)
{
    bufferTransition(cb, this->handle, this->state, this->byteSize, newState);
}

vk::BufferMemoryBarrier2 TexelBuffer::transitionBarrier(
    const BufferState &newState)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->size, newState);
}

void TexelBuffer::transition(
    const vk::CommandBuffer cb, const BufferState &newState)
{
    bufferTransition(cb, this->handle, this->state, this->size, newState);
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
