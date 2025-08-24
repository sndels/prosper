#include "GBuffer.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "wheels/assert.hpp"

namespace render
{

void GBuffer::create(const vk::Extent2D &extent)
{
    WHEELS_ASSERT(!albedoRoughness.isValid());
    WHEELS_ASSERT(!normalMetalness.isValid());
    WHEELS_ASSERT(!velocity.isValid());
    WHEELS_ASSERT(!depth.isValid());

    albedoRoughness = gRenderResources.images->create(
        gfx::ImageDescription{
            .format = sAlbedoRoughnessFormat,
            .width = extent.width,
            .height = extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |         // Debug
                          vk::ImageUsageFlagBits::eColorAttachment | // Render
                          vk::ImageUsageFlagBits::eStorage,          // Shading
        },
        "albedoRoughness");
    normalMetalness = gRenderResources.images->create(
        gfx::ImageDescription{
            .format = sNormalMetalnessFormat,
            .width = extent.width,
            .height = extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eSampled |         // Debug
                          vk::ImageUsageFlagBits::eColorAttachment | // Render
                          vk::ImageUsageFlagBits::eStorage,          // Shading
        },
        "normalMetalness");
    velocity = createVelocity(extent, "velocity");
    depth = createDepth(extent, "depth");
}

void GBuffer::setHistoryDebugNames() const
{
    if (!gRenderResources.images->isValidHandle(albedoRoughness))
        return;

    WHEELS_ASSERT(gRenderResources.images->isValidHandle(normalMetalness));
    WHEELS_ASSERT(gRenderResources.images->isValidHandle(depth));

    gRenderResources.images->appendDebugName(
        albedoRoughness, "previousAlbedoRoughness");
    gRenderResources.images->appendDebugName(
        normalMetalness, "previousNormalMetallic");
    // Skip velocity history as no one needs it
    gRenderResources.images->appendDebugName(depth, "previousDepth");
}

void GBuffer::releaseAll() const
{
    if (!gRenderResources.images->isValidHandle(albedoRoughness))
        return;

    WHEELS_ASSERT(gRenderResources.images->isValidHandle(normalMetalness));
    WHEELS_ASSERT(gRenderResources.images->isValidHandle(depth));

    gRenderResources.images->release(albedoRoughness);
    gRenderResources.images->release(normalMetalness);
    // Velocity is not present in history gbuffer
    if (gRenderResources.images->isValidHandle(velocity))
        gRenderResources.images->release(velocity);
    gRenderResources.images->release(depth);
}

void GBuffer::preserveAll() const
{
    gRenderResources.images->preserve(albedoRoughness);
    gRenderResources.images->preserve(normalMetalness);
    // Skip velocity history as no one needs it
    gRenderResources.images->preserve(depth);
}

} // namespace render
