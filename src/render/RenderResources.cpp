#include "RenderResources.hpp"

#include "gfx/Device.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/allocators/utils.hpp>

using namespace wheels;

// This is used everywhere and init()/destroy() order relative to other similar
// globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RenderResources gRenderResources;

RenderResources::~RenderResources()
{
    WHEELS_ASSERT(
        (!m_initialized || nearestSampler == vk::Sampler{}) &&
        "destroy() not called?");
}

void RenderResources::init()
{
    this->images = OwningPtr<RenderImageCollection>(gAllocators.general);
    this->buffers = OwningPtr<RenderBufferCollection>(gAllocators.general);
    this->texelBuffers =
        OwningPtr<RenderTexelBufferCollection>(gAllocators.general);

    this->nearestBorderBlackFloatSampler =
        gDevice.logical().createSampler(vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToBorder,
            .addressModeV = vk::SamplerAddressMode::eClampToBorder,
            .addressModeW = vk::SamplerAddressMode::eClampToBorder,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
        });
    this->nearestSampler =
        gDevice.logical().createSampler(vk::SamplerCreateInfo{
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
    this->bilinearSampler =
        gDevice.logical().createSampler(vk::SamplerCreateInfo{
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
    this->trilinearSampler =
        gDevice.logical().createSampler(vk::SamplerCreateInfo{
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

    m_initialized = true;
}

void RenderResources::destroy()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.logical().destroy(nearestBorderBlackFloatSampler);
    gDevice.logical().destroy(nearestSampler);
    gDevice.logical().destroy(bilinearSampler);
    gDevice.logical().destroy(trilinearSampler);

    nearestSampler = vk::Sampler{};
    bilinearSampler = vk::Sampler{};
    trilinearSampler = vk::Sampler{};

    images.reset();
    texelBuffers.reset();
    buffers.reset();
}

void RenderResources::startFrame()
{
    WHEELS_ASSERT(m_initialized);

    images->startFrame();
    texelBuffers->startFrame();
    buffers->startFrame();
}

void RenderResources::destroyResources()
{
    WHEELS_ASSERT(m_initialized);

    images->destroyResources();
    texelBuffers->destroyResources();
    buffers->destroyResources();
}

void transition(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const Transitions &transitions)
{
    wheels::Array<vk::ImageMemoryBarrier2> imageBarriers{
        scopeAlloc, transitions.images.size()};
    for (const auto &image_state : transitions.images)
    {
        const wheels::Optional<vk::ImageMemoryBarrier2> barrier =
            gRenderResources.images->transitionBarrier(
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
            gRenderResources.buffers->transitionBarrier(
                buffer_state.first, buffer_state.second);
        if (barrier.has_value())
            bufferBarriers.push_back(*barrier);
    }
    for (const auto &buffer_state : transitions.texelBuffers)
    {
        const wheels::Optional<vk::BufferMemoryBarrier2> barrier =
            gRenderResources.texelBuffers->transitionBarrier(
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
