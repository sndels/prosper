#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "DebugGeometry.hpp"
#include "DescriptorAllocator.hpp"
#include "Device.hpp"
#include "RenderImageCollection.hpp"
#include "RenderResourceCollection.hpp"
#include "Utils.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

using BufferHandle = RenderResourceHandle<Buffer>;
using TexelBufferHandle = RenderResourceHandle<TexelBuffer>;

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

    // Have this be static because ImGuiRenderer uses it in its framebuffer.
    // Don't want to reallocate FBs each frame if this ends up ping-ponging with
    // some other resource
    Image finalComposite;

    vk::Sampler nearestSampler;
    vk::Sampler bilinearSampler;

    // One lines buffer per frame to leave mapped
    wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
