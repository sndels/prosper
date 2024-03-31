#include "RenderResources.hpp"

#include <wheels/allocators/utils.hpp>

using namespace wheels;

RenderResources::RenderResources(wheels::Allocator &alloc) noexcept
: images{alloc}
, texelBuffers{alloc}
, buffers{alloc}
{
}

RenderResources::~RenderResources()
{
    if (device != nullptr)
    {
        device->logical().destroy(nearestSampler);
        device->logical().destroy(bilinearSampler);
        device->logical().destroy(trilinearSampler);
    }
}

void RenderResources::init(Device *d)
{
    WHEELS_ASSERT(device == nullptr);
    WHEELS_ASSERT(d != nullptr);

    device = d;
    images.init(device);
    texelBuffers.init(device);
    buffers.init(device);
    constantsRing.init(
        device, vk::BufferUsageFlagBits::eStorageBuffer,
        asserted_cast<uint32_t>(kilobytes(16)), "ConstantsRing");

    this->nearestSampler =
        device->logical().createSampler(vk::SamplerCreateInfo{
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
    trilinearSampler = device->logical().createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    });
}

void RenderResources::startFrame()
{
    WHEELS_ASSERT(device != nullptr);

    images.startFrame();
    texelBuffers.startFrame();
    buffers.startFrame();
    constantsRing.startFrame();
}

void RenderResources::destroyResources()
{
    WHEELS_ASSERT(device != nullptr);

    images.destroyResources();
    texelBuffers.destroyResources();
    buffers.destroyResources();
}

void transition(
    wheels::ScopedScratch scopeAlloc, RenderResources &resources,
    vk::CommandBuffer cb, const Transitions &transitions)
{
    wheels::Array<vk::ImageMemoryBarrier2> imageBarriers{
        scopeAlloc, transitions.images.size()};
    for (const auto &image_state : transitions.images)
    {
        const wheels::Optional<vk::ImageMemoryBarrier2> barrier =
            resources.images.transitionBarrier(
                image_state.first, image_state.second);
        if (barrier.has_value())
            imageBarriers.push_back(*barrier);
    }

    wheels::Array<vk::BufferMemoryBarrier2> bufferBarriers{
        scopeAlloc,
        transitions.buffers.size() + transitions.texelBuffers.size()};
    for (const auto &buffer_state : transitions.buffers)
    {
        const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
            resources.buffers.transitionBarrier(
                buffer_state.first, buffer_state.second);
        if (barrier.has_value())
            bufferBarriers.push_back(*barrier);
    }
    for (const auto &buffer_state : transitions.texelBuffers)
    {
        const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
            resources.texelBuffers.transitionBarrier(
                buffer_state.first, buffer_state.second);
        if (barrier.has_value())
            bufferBarriers.push_back(*barrier);
    }

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}
