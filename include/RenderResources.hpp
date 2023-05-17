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

// Renderpasses that create the resources are responsible for their recreation,
// lifetime
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

    struct Images
    {
        Image sceneColor;
        Image sceneDepth;
        Image toneMapped;
        Image albedoRoughness;
        Image normalMetalness;
        Image finalComposite;
    };

    struct Buffers
    {
        struct
        {
            Image pointers;
            TexelBuffer indicesCount;
            TexelBuffer indices;
            vk::DescriptorSetLayout descriptorSetLayout;
            wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
                descriptorSets{{}};
        } lightClusters;
        // One lines buffer per swap image to leave mapped
        wheels::StaticArray<DebugLines, MAX_FRAMES_IN_FLIGHT> debugLines;
    };

    // Both alloc and device need to live as long as this
    RenderResources(wheels::Allocator &alloc, Device *device)
    : staticDescriptorsAlloc{alloc, device}
    , images{alloc, device}
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

    // This allocator should only be used for the descriptors that can live
    // until the end of the program. As such, reset() shouldn't be called so
    // that users can rely on the descriptors being there once allocated.
    // TODO: Does this have to be here? Shouldn't this only be used to construct
    // things up front?
    DescriptorAllocator staticDescriptorsAlloc;
    Images staticImages;
    Buffers staticBuffers;

    RenderImageCollection images;
    RenderTexelBufferCollection texelBuffers;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
