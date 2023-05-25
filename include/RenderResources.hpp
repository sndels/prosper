#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "DebugGeometry.hpp"
#include "DescriptorAllocator.hpp"
#include "Device.hpp"
#include "RenderResourceCollection.hpp"
#include "Utils.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

using TexelBufferHandle = RenderResourceHandle<TexelBuffer>;
using ImageHandle = RenderResourceHandle<Image>;

class RenderResources
{
  public:
    using RenderTexelBufferCollection = RenderResourceCollection<
        TexelBufferHandle, TexelBuffer, TexelBufferDescription,
        TexelBufferCreateInfo, BufferState, vk::BufferMemoryBarrier2,
        vk::Buffer, VkBuffer, vk::ObjectType::eBuffer>;
    using RenderImageCollection = RenderResourceCollection<
        ImageHandle, Image, ImageDescription, ImageCreateInfo, ImageState,
        vk::ImageMemoryBarrier2, vk::Image, VkImage, vk::ObjectType::eImage>;

    // Both alloc and device need to live as long as this
    RenderResources(wheels::Allocator &alloc, Device *device)
    : images{alloc, device}
    , texelBuffers{alloc, device}
    {
    }
    ~RenderResources() = default;

    RenderResources(RenderResources &) = delete;
    RenderResources(RenderResources &&) = delete;
    RenderResources &operator=(RenderResources &) = delete;
    RenderResources &operator=(RenderResources &&) = delete;

    // Should be called at the start of the frame so resources will get the
    // correct names set
    void clearDebugNames()
    {
        images.clearDebugNames();
        texelBuffers.clearDebugNames();
    }

    // Should be called e.g. when viewport is resized since the render resources
    // will be created with different sizes on the next frame
    void destroyResources()
    {
        images.destroyResources();
        texelBuffers.destroyResources();
    }

    RenderImageCollection images;
    RenderTexelBufferCollection texelBuffers;

    // Have this be static because ImGuiRenderer uses it in its framebuffer.
    // Don't want to reallocate FBs each frame if this ends up ping-ponging with
    // some other resource
    Image finalComposite;

    // One lines buffer per frame to leave mapped
    wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
