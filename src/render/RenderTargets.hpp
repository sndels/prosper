#ifndef PROSPER_RENDER_TARGETS_HPP
#define PROSPER_RENDER_TARGETS_HPP

#include "RenderResources.hpp"

const vk::Format sDepthFormat = vk::Format::eD32Sfloat;
const vk::Format sIlluminationFormat = vk::Format::eR16G16B16A16Sfloat;

[[nodiscard]] ImageHandle createDepth(
    Device &device, RenderResources &resources, const vk::Extent2D &size,
    const char *debugName);

[[nodiscard]] ImageHandle createIllumination(
    RenderResources &resources, const vk::Extent2D &size,
    const char *debugName);

#endif // PROSPER_RENDER_TARGETS_HPP