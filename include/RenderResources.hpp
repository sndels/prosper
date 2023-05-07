#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "DebugGeometry.hpp"
#include "DescriptorAllocator.hpp"
#include "Device.hpp"
#include "Utils.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/static_array.hpp>

// Renderpasses that create the resources are responsible for their recreation,
// lifetime
struct RenderResources
{
    struct Images
    {
        Image sceneColor;
        Image sceneDepth;
        Image toneMapped;
        Image albedoRoughness;
        Image normalMetalness;
    };

    struct Buffers
    {
        struct
        {
            Image pointers;
            TexelBuffer indicesCount;
            TexelBuffer indices;
            vk::DescriptorSetLayout descriptorSetLayout;
            wheels::StaticArray<vk::DescriptorSet, MAX_SWAPCHAIN_IMAGES>
                descriptorSets;
        } lightClusters;
        // One lines buffer per swap image to leave mapped
        wheels::StaticArray<DebugLines, MAX_SWAPCHAIN_IMAGES> debugLines;
    };

    // Both alloc and device need to live as long as this
    RenderResources(wheels::Allocator &alloc, Device *device)
    : descriptorAllocator{alloc, device}
    {
    }

    // Pools will be reset on swapchain recreation as many passes have resources
    // tied to swapchain (resolution) and it is clearer to recreate everything
    // instead of cherry-picking just the descriptors that need to be updated
    DescriptorAllocator descriptorAllocator;
    Images images;
    Buffers buffers;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
