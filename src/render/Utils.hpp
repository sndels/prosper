#ifndef PROSPER_RENDER_UTILS_HPP
#define PROSPER_RENDER_UTILS_HPP

#include "render/RenderResourceHandle.hpp"

#include <vulkan/vulkan.hpp>

namespace render
{

vk::Extent2D getExtent2D(ImageHandle image);
vk::Extent2D getRoundedUpHalfExtent2D(ImageHandle image);
vk::Rect2D getRect2D(ImageHandle image);

} // namespace render

#endif // PROSPER_RENDER_UTILS_HPP
