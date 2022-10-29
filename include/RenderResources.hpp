#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "DescriptorAllocator.hpp"
#include "Device.hpp"


// Renderpasses that create the resources are responsible for their recreation,
// lifetime
struct RenderResources
{
    struct Images
    {
        Image sceneColor;
        Image sceneDepth;
        Image toneMapped;
    };

    struct Buffers
    {
        struct
        {
            Image pointers;
            TexelBuffer indicesCount;
            TexelBuffer indices;
            vk::DescriptorSetLayout descriptorSetLayout;
            std::vector<vk::DescriptorSet> descriptorSets;
        } lightClusters;
    };

    RenderResources(Device *device)
    : descriptorAllocator{device}
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
