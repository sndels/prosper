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
    RenderResources(wheels::Allocator &alloc) noexcept;
    ~RenderResources();

    RenderResources(RenderResources &) = delete;
    RenderResources(RenderResources &&) = delete;
    RenderResources &operator=(RenderResources &) = delete;
    RenderResources &operator=(RenderResources &&) = delete;

    void init(Device *d);

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

    vk::Sampler nearestSampler;
    vk::Sampler bilinearSampler;
    vk::Sampler trilinearSampler;

    // One lines buffer per frame to leave mapped
    wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;
};

struct Transitions
{
    wheels::Span<const ImageTransition> images;
    wheels::Span<const BufferTransition> buffers;
    wheels::Span<const TexelBufferTransition> texelBuffers;
};
void transition(
    wheels::ScopedScratch scopeAlloc, RenderResources &resources,
    vk::CommandBuffer cb, const Transitions &transitions);

#endif // PROSPER_RENDER_RESOURCES_HPP
