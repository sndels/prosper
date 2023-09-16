#ifndef PROSPER_SCENE_DEBUG_GEOMETRY_HPP
#define PROSPER_SCENE_DEBUG_GEOMETRY_HPP

#include "../gfx/Resources.hpp"

#include <glm/glm.hpp>

struct DebugLines
{
    // Writing more than 100k lines per frame sounds slow
    static const vk::DeviceSize sMaxLineCount = 100'000;
    // A line is two positions and a color
    static const vk::DeviceSize sLineBytes = sizeof(float) * 9;
    Buffer buffer;
    uint32_t count{0};

    void reset();
    void addLine(glm::vec3 p0, glm::vec3 p1, glm::vec3 color);
};

#endif // PROSPER_SCENE_DEBUG_GEOMETRY_HPP
