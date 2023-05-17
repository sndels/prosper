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
    {
    }

    // This allocator should only be used for the descriptors that can live
    // until the end of the program. As such, reset() shouldn't be called so
    // that users can rely on the descriptors being there once allocated.
    DescriptorAllocator staticDescriptorsAlloc;
    Images staticImages;
    Buffers staticBuffers;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
