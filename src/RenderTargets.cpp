#include "RenderTargets.hpp"

ImageHandle createDepth(
    Device &device, RenderResources &resources, const vk::Extent2D &size,
    const char *debugName)
{
    // Check depth buffer without stencil is supported
    const auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
    const auto properties = device.physical().getFormatProperties(sDepthFormat);
    if ((properties.optimalTilingFeatures & features) != features)
        throw std::runtime_error("Depth format unsupported");

    return resources.images.create(
        ImageDescription{
            .format = sDepthFormat,
            .width = size.width,
            .height = size.height,
            .usageFlags =
                vk::ImageUsageFlagBits::eDepthStencilAttachment | // Geometry
                vk::ImageUsageFlagBits::eSampled, // Deferred shading
        },
        debugName);
}

ImageHandle createIllumination(
    RenderResources &resources, const vk::Extent2D &size, const char *debugName)
{
    return resources.images.create(
        ImageDescription{
            .format = sIlluminationFormat,
            .width = size.width,
            .height = size.height,
            .usageFlags = vk::ImageUsageFlagBits::eColorAttachment | // Render
                          vk::ImageUsageFlagBits::eStorage,          // ToneMap
        },
        debugName);
}
