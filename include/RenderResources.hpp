#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

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

    struct DescriptorPools
    {
        vk::DescriptorPool constant;
        vk::DescriptorPool swapchainRelated;
    };

    DescriptorPools descriptorPools;
    Images images;
    Buffers buffers;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
