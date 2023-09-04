#include "Resources.hpp"

namespace
{

template <typename State>
vk::PipelineStageFlags2 nativeStagesCommon(State state)
{

    vk::PipelineStageFlags2 flags;

    if (contains(state, State::StageFragmentShader))
        flags |= vk::PipelineStageFlagBits2::eFragmentShader;
    if (contains(state, State::StageComputeShader))
        flags |= vk::PipelineStageFlagBits2::eComputeShader;
    if (contains(state, State::StageTransfer))
        flags |= vk::PipelineStageFlagBits2::eTransfer;

    return flags;
}

vk::PipelineStageFlags2 nativeStages(BufferState state)
{
    return nativeStagesCommon(state);
}

vk::PipelineStageFlags2 nativeStages(ImageState state)
{

    vk::PipelineStageFlags2 flags = nativeStagesCommon(state);

    if (contains(state, ImageState::StageEarlyFragmentTests))
        flags |= vk::PipelineStageFlagBits2::eEarlyFragmentTests;
    if (contains(state, ImageState::StageLateFragmentTests))
        flags |= vk::PipelineStageFlagBits2::eLateFragmentTests;
    if (contains(state, ImageState::StageColorAttachmentOutput))
        flags |= vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    if (contains(state, ImageState::StageRayTracingShader))
        flags |= vk::PipelineStageFlagBits2::eRayTracingShaderKHR;

    return flags;
}

template <typename State> vk::AccessFlags2 nativeAccessesCommon(State state)
{

    vk::AccessFlags2 flags;

    if (contains(state, State::AccessShaderRead))
        flags |= vk::AccessFlagBits2::eShaderRead;
    if (contains(state, State::AccessShaderWrite))
        flags |= vk::AccessFlagBits2::eShaderWrite;
    if (contains(state, State::AccessTransferRead))
        flags |= vk::AccessFlagBits2::eTransferRead;
    if (contains(state, State::AccessTransferWrite))
        flags |= vk::AccessFlagBits2::eTransferWrite;

    return flags;
}

vk::AccessFlags2 nativeAccesses(BufferState state)
{
    return nativeAccessesCommon(state);
}

vk::AccessFlags2 nativeAccesses(ImageState state)
{

    vk::AccessFlags2 flags = nativeAccessesCommon(state);

    if (contains(state, ImageState::AccessColorAttachmentRead))
        flags |= vk::AccessFlagBits2::eColorAttachmentRead;
    if (contains(state, ImageState::AccessColorAttachmentWrite))
        flags |= vk::AccessFlagBits2::eColorAttachmentWrite;
    if (contains(state, ImageState::AccessDepthAttachmentRead))
        flags |= vk::AccessFlagBits2::eDepthStencilAttachmentRead;
    if (contains(state, ImageState::AccessDepthAttachmentWrite))
        flags |= vk::AccessFlagBits2::eDepthStencilAttachmentRead;

    return flags;
}

vk::ImageLayout nativeLayout(ImageState state)
{
    if (contains(state, ImageState::AccessShaderRead) ||
        contains(state, ImageState::AccessShaderWrite))
        return vk::ImageLayout::eGeneral;
    if (contains(state, ImageState::StageColorAttachmentOutput))
        return vk::ImageLayout::eColorAttachmentOptimal;
    if (contains(state, ImageState::AccessDepthAttachmentWrite))
        return vk::ImageLayout::eDepthAttachmentOptimal;
    if (contains(state, ImageState::AccessDepthAttachmentRead))
        // Write was above so no write flag present
        return vk::ImageLayout::eDepthReadOnlyOptimal;
    if (contains(state, ImageState::AccessTransferRead))
        return vk::ImageLayout::eTransferSrcOptimal;
    if (contains(state, ImageState::AccessTransferWrite))
        return vk::ImageLayout::eTransferDstOptimal;

    assert(state == ImageState::Unknown);

    return vk::ImageLayout::eUndefined;
}

vk::BufferMemoryBarrier2 bufferTransitionBarrier(
    vk::Buffer &buffer, BufferState &currentState, vk::DeviceSize size,
    BufferState newState)
{
    // TODO:
    // Skip barriers when states are the same
    // Only include current write stages (and accesses) if Write->Read

    const vk::PipelineStageFlags2 srcStageMask = nativeStages(currentState);
    const vk::AccessFlags2 srcAccessMask = nativeAccesses(currentState);

    const vk::PipelineStageFlags2 dstStageMask = nativeStages(newState);
    const vk::AccessFlags2 dstAccessMask = nativeAccesses(newState);

    const vk::BufferMemoryBarrier2 barrier{
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = 0,
        .size = size,
    };

    currentState = newState;

    return barrier;
}

void bufferTransition(
    const vk::CommandBuffer cb, vk::Buffer &buffer, BufferState &currentState,
    vk::DeviceSize size, BufferState newState)

{
    const vk::BufferMemoryBarrier2 barrier =
        bufferTransitionBarrier(buffer, currentState, size, newState);
    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });
}

} // namespace

vk::BufferMemoryBarrier2 Buffer::transitionBarrier(BufferState newState)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->byteSize, newState);
}

void Buffer::transition(const vk::CommandBuffer cb, BufferState newState)
{
    bufferTransition(cb, this->handle, this->state, this->byteSize, newState);
}

vk::BufferMemoryBarrier2 TexelBuffer::transitionBarrier(BufferState newState)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->size, newState);
}

void TexelBuffer::transition(const vk::CommandBuffer cb, BufferState newState)
{
    bufferTransition(cb, this->handle, this->state, this->size, newState);
}

vk::ImageMemoryBarrier2 Image::transitionBarrier(ImageState newState)
{
    // TODO:
    // Skip barriers when states are the same
    // Only include current write stages (and accesses) if Write->Read

    const vk::PipelineStageFlags2 srcStageMask = nativeStages(this->state);
    const vk::AccessFlags2 srcAccessMask = nativeAccesses(this->state);

    const vk::PipelineStageFlags2 dstStageMask = nativeStages(newState);
    const vk::AccessFlags2 dstAccessMask = nativeAccesses(newState);

    const vk::ImageLayout oldLayout = nativeLayout(this->state);
    const vk::ImageLayout newLayout = nativeLayout(newState);

    // TODO:
    // If current access includes depth write, should use stage mask late frag
    // tests as source ?
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresourceRange = subresourceRange,
    };

    this->state = newState;

    return barrier;
}

void Image::transition(const vk::CommandBuffer buffer, ImageState newState)
{
    const vk::ImageMemoryBarrier2 barrier = transitionBarrier(newState);
    buffer.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}
