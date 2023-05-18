#include "VkUtils.hpp"

void setViewportScissor(vk::CommandBuffer cb, const vk::Rect2D &area)
{
    const vk::Viewport viewport{
        .x = static_cast<float>(area.offset.x),
        .y = static_cast<float>(area.offset.y),
        .width = static_cast<float>(area.extent.width),
        .height = static_cast<float>(area.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    cb.setViewport(0, 1, &viewport);
    cb.setScissor(0, 1, &area);
}
