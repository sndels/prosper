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

template <typename State> bool hasWriteAccessesCommon(State state)
{
    if (contains(state, State::AccessShaderWrite))
        return true;
    if (contains(state, State::AccessTransferWrite))
        return true;

    return false;
}

bool hasWriteAccesses(BufferState state)
{
    return hasWriteAccessesCommon(state);
}

bool hasWriteAccesses(ImageState state)
{
    if (hasWriteAccessesCommon(state))
        return true;

    if (contains(state, ImageState::AccessColorAttachmentWrite))
        return true;
    if (contains(state, ImageState::AccessDepthAttachmentWrite))
        return true;

    return false;
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

wheels::Optional<vk::BufferMemoryBarrier2> bufferTransitionBarrier(
    vk::Buffer &buffer, BufferState &currentState, vk::DeviceSize size,
    BufferState newState, bool force_barrier = false)
{
    // TODO:
    // Use memory barriers instead of buffer barriers as queue transitions
    // aren't needed

    // Skip redundant barriers
    // NOTE:
    // We can't skip when stages differ because execution dependency might be
    // missing for the new mask.
    //
    // Consider a write-read barrier with layout transition into fragment shader
    // read access followed by a compute shader read access: even if both use
    // the same layout, compute has to wait for the original write. If the
    // layout transition doesn't include compute in dstStageMask (which we can't
    // assume), we need an execution dependency between the frag and compute.
    //
    // This could be avoided if we had global (graph) information and could move
    // the compute stage bit to the barrier before the frag pass. It would also
    // allow the driver to overlap the two reading passes.
    if (!force_barrier && currentState == newState &&
        !hasWriteAccesses(currentState) && !hasWriteAccesses(newState))
        return {};

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
    const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
        bufferTransitionBarrier(buffer, currentState, size, newState);
    if (barrier.has_value())
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &*barrier,
        });
}

} // namespace

wheels::Optional<vk::BufferMemoryBarrier2> Buffer::transitionBarrier(
    BufferState newState, bool force_barrier)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->byteSize, newState, force_barrier);
}

void Buffer::transition(const vk::CommandBuffer cb, BufferState newState)
{
    bufferTransition(cb, this->handle, this->state, this->byteSize, newState);
}

wheels::Optional<vk::BufferMemoryBarrier2> TexelBuffer::transitionBarrier(
    BufferState newState, bool force_barrier)
{
    return bufferTransitionBarrier(
        this->handle, this->state, this->size, newState, force_barrier);
}

void TexelBuffer::transition(const vk::CommandBuffer cb, BufferState newState)
{
    bufferTransition(cb, this->handle, this->state, this->size, newState);
}

wheels::Optional<vk::ImageMemoryBarrier2> Image::transitionBarrier(
    ImageState newState, bool force_barrier)
{
    // TODO:
    // Use memory barriers instead when layout change isn't required

    // Skip read-read barriers with matching stages that don't change layouts.
    // NOTE:
    // We can't skip when stages differ because execution dependency might be
    // missing for the new mask.
    //
    // Consider a write-read barrier with layout transition into fragment shader
    // read access followed by a compute shader read access: even if both use
    // the same layout, compute has to wait for the original write. If the
    // layout transition doesn't include compute in dstStageMask (which we can't
    // assume), we need an execution dependency between the frag and compute.
    //
    // This could be avoided if we had global (graph) information and could move
    // the compute stage bit to the barrier before the frag pass. It would also
    // allow the driver to overlap the two reading passes.
    if (!force_barrier && this->state == newState &&
        !hasWriteAccesses(this->state))
        return {};

    const vk::ImageLayout oldLayout = nativeLayout(this->state);
    const vk::ImageLayout newLayout = nativeLayout(newState);

    const vk::PipelineStageFlags2 srcStageMask = nativeStages(this->state);
    const vk::AccessFlags2 srcAccessMask = nativeAccesses(this->state);

    const vk::PipelineStageFlags2 dstStageMask = nativeStages(newState);
    const vk::AccessFlags2 dstAccessMask = nativeAccesses(newState);

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
    const wheels::Optional<vk::ImageMemoryBarrier2> barrier =
        transitionBarrier(newState);
    if (barrier.has_value())
        buffer.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &*barrier,
        });
}
