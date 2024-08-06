#ifndef PROSPER_RENDER_TARGETS_HPP
#define PROSPER_RENDER_TARGETS_HPP

#include "RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>

const vk::Format sDepthFormat = vk::Format::eD32Sfloat;
const vk::Format sIlluminationFormat = vk::Format::eR16G16B16A16Sfloat;
const vk::Format sVelocityFormat = vk::Format::eR16G16Snorm;
const vk::Format sFinalCompositeFormat = vk::Format::eR8G8B8A8Unorm;

[[nodiscard]] ImageHandle createDepth(
    const vk::Extent2D &size, const char *debugName);

[[nodiscard]] ImageHandle createIllumination(
    const vk::Extent2D &size, const char *debugName);

[[nodiscard]] ImageHandle createVelocity(
    const vk::Extent2D &size, const char *debugName);

#endif // PROSPER_RENDER_TARGETS_HPP
