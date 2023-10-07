#include "RenderResources.hpp"

#include <wheels/allocators/utils.hpp>

using namespace wheels;

RenderResources::RenderResources(wheels::Allocator &alloc, Device *device)
: device{device}
, images{alloc, device}
, texelBuffers{alloc, device}
, buffers{alloc, device}
, constantsRing{
      device, vk::BufferUsageFlagBits::eStorageBuffer,
      asserted_cast<uint32_t>(kilobytes(16)), "ConstantsRing"}
{
    assert(device != nullptr);

    nearestSampler = device->logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    });
    bilinearSampler = device->logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    });
}

RenderResources::~RenderResources()
{
    if (device != nullptr)
    {
        device->logical().destroy(nearestSampler);
        device->logical().destroy(bilinearSampler);
    }
}

void RenderResources::startFrame()
{
    images.startFrame();
    texelBuffers.startFrame();
    buffers.startFrame();
    constantsRing.startFrame();
}

void RenderResources::destroyResources()
{
    images.destroyResources();
    texelBuffers.destroyResources();
    buffers.destroyResources();
}
