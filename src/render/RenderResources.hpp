#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "../gfx/Resources.hpp"
#include "../gfx/RingBuffer.hpp"
#include "../scene/DebugGeometry.hpp"
#include "../utils/Utils.hpp"
#include "RenderImageCollection.hpp"
#include "RenderResourceCollection.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/concepts.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>

using ImageTransition = wheels::Pair<ImageHandle, ImageState>;
using BufferTransition = wheels::Pair<BufferHandle, BufferState>;
using TexelBufferTransition = wheels::Pair<TexelBufferHandle, BufferState>;

class RenderResources
{
  public:
    using RenderBufferCollection = RenderResourceCollection<
        BufferHandle, Buffer, BufferDescription, BufferCreateInfo, BufferState,
        vk::BufferMemoryBarrier2, vk::Buffer, VkBuffer,
        vk::ObjectType::eBuffer>;
    using RenderTexelBufferCollection = RenderResourceCollection<
        TexelBufferHandle, TexelBuffer, TexelBufferDescription,
        TexelBufferCreateInfo, BufferState, vk::BufferMemoryBarrier2,
        vk::Buffer, VkBuffer, vk::ObjectType::eBuffer>;

    // Both alloc and device need to live as long as this
    RenderResources(wheels::Allocator &alloc, Device *device);
    ~RenderResources();

    RenderResources(RenderResources &) = delete;
    RenderResources(RenderResources &&) = delete;
    RenderResources &operator=(RenderResources &) = delete;
    RenderResources &operator=(RenderResources &&) = delete;

    // Should be called at the start of the frame so resources will get the
    // correct names set
    void startFrame();

    // Should be called e.g. when viewport is resized since the render resources
    // will be created with different sizes on the next frame
    void destroyResources();

    Device *device{nullptr};

    RenderImageCollection images;
    RenderTexelBufferCollection texelBuffers;
    RenderBufferCollection buffers;

    RingBuffer constantsRing;

    // Have this be static because ImGuiRenderer uses it in its framebuffer.
    // Don't want to reallocate FBs each frame if this ends up ping-ponging with
    // some other resource
    Image finalComposite;

    vk::Sampler nearestSampler;
    vk::Sampler bilinearSampler;

    // One lines buffer per frame to leave mapped
    wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;
};

// TODO:
// Is there a less messy way to template this while only allowing the exact
// input types?
template <
    template <class, size_t> class ImageArray, size_t ImageCount,
    template <class, size_t> class BufferArray, size_t BufferCount,
    template <class, size_t> class TexelBufferArray, size_t TexelBufferCount>
    requires(
        (wheels::SameAs<
             ImageArray<ImageTransition, ImageCount>,
             wheels::StaticArray<ImageTransition, ImageCount>> ||
         wheels::SameAs<
             ImageArray<ImageTransition, ImageCount>,
             wheels::InlineArray<ImageTransition, ImageCount>>) &&
        (wheels::SameAs<
             BufferArray<BufferTransition, BufferCount>,
             wheels::StaticArray<BufferTransition, BufferCount>> ||
         wheels::SameAs<
             BufferArray<BufferTransition, BufferCount>,
             wheels::InlineArray<BufferTransition, BufferCount>>) &&
        (wheels::SameAs<
             TexelBufferArray<TexelBufferTransition, TexelBufferCount>,
             wheels::StaticArray<TexelBufferTransition, TexelBufferCount>> ||
         wheels::SameAs<
             TexelBufferArray<TexelBufferTransition, TexelBufferCount>,
             wheels::InlineArray<TexelBufferTransition, TexelBufferCount>>))
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const ImageArray<ImageTransition, ImageCount> &images,
    const BufferArray<BufferTransition, BufferCount> &buffers,
    const TexelBufferArray<TexelBufferTransition, TexelBufferCount>
        &texelBuffers)
{
    // TODO:
    // The current implementation with "overloads" of subsets of the inputs
    // allocates stack space for a minimum 1 transition and 1 barrier per
    // resource type. Is a tighter implementation possible while keeping the
    // ergonomics and not having implicit allocators for heap containers?

    wheels::InlineArray<vk::ImageMemoryBarrier2, ImageCount> imageBarriers;
    for (const auto &image_state : images)
    {
        const wheels::Optional<vk::ImageMemoryBarrier2> barrier =
            resources.images.transitionBarrier(
                image_state.first, image_state.second);
        if (barrier.has_value())
            imageBarriers.push_back(*barrier);
    }

    wheels::InlineArray<
        vk::BufferMemoryBarrier2, BufferCount + TexelBufferCount>
        bufferBarriers;
    for (const auto &buffer_state : buffers)
    {
        const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
            resources.buffers.transitionBarrier(
                buffer_state.first, buffer_state.second);
        if (barrier.has_value())
            bufferBarriers.push_back(*barrier);
    }
    for (const auto &buffer_state : texelBuffers)
    {
        const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
            resources.texelBuffers.transitionBarrier(
                buffer_state.first, buffer_state.second);
        if (barrier.has_value())
            bufferBarriers.push_back(*barrier);
    }

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}

template <size_t ImageCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<ImageTransition, ImageCount> &images)
{
    const wheels::InlineArray<BufferTransition, 1> buffers;
    const wheels::InlineArray<TexelBufferTransition, 1> texelBuffers;
    transition(resources, cb, images, buffers, texelBuffers);
}

template <size_t BufferCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<BufferTransition, BufferCount> &buffers)
{
    const wheels::InlineArray<ImageTransition, 1> images;
    const wheels::InlineArray<TexelBufferTransition, 1> texelBuffers;
    transition(resources, cb, images, buffers, texelBuffers);
}

template <size_t TexelBufferCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<TexelBufferTransition, TexelBufferCount>
        &texelBuffers)
{
    const wheels::InlineArray<ImageTransition, 1> images;
    const wheels::InlineArray<BufferTransition, 1> buffers;
    transition(resources, cb, images, buffers, texelBuffers);
}

template <size_t ImageCount, size_t BufferCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<ImageTransition, ImageCount> &images,
    const wheels::StaticArray<BufferTransition, BufferCount> &buffers)
{
    const wheels::InlineArray<TexelBufferTransition, 1> texelBuffers;
    transition(resources, cb, images, buffers, texelBuffers);
}

template <size_t ImageCount, size_t TexelBufferCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<ImageTransition, ImageCount> &images,
    const wheels::StaticArray<TexelBufferTransition, TexelBufferCount>
        &texelBuffers)
{
    const wheels::InlineArray<BufferTransition, 1> buffers;
    transition(resources, cb, images, buffers, texelBuffers);
}

template <size_t BufferCount, size_t TexelBufferCount>
void transition(
    RenderResources &resources, vk::CommandBuffer cb,
    const wheels::StaticArray<BufferTransition, BufferCount> &buffers,
    const wheels::StaticArray<TexelBufferTransition, TexelBufferCount>
        &texelBuffers)
{
    const wheels::InlineArray<ImageTransition, 1> images;
    transition(resources, cb, images, buffers, texelBuffers);
}

#endif // PROSPER_RENDER_RESOURCES_HPP
