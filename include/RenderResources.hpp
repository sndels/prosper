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

    struct DescriptorPools
    {
        vk::DescriptorPool constant;
        vk::DescriptorPool swapchainRelated;
    };

    DescriptorPools descriptorPools;
    Images images;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
