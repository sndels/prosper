#ifndef PROSPER_DEBUG_GEOMETRY_HPP
#define PROSPER_DEBUG_GEOMETRY_HPP

#include "Resources.hpp"

#include <glm/glm.hpp>

struct DebugLines
{
    // Writing more than 100k lines per frame sounds slow
    static const uint32_t sMaxLineCount = 100'000;
    // A line is two positions and a color
    static const uint32_t sLineBytes = sizeof(float) * 9;
    Buffer buffer;
    uint32_t count{0};

    void reset();
    void addLine(glm::vec3 p0, glm::vec3 p1, glm::vec3 color);
};

#endif // PROSPER_DEBUG_GEOMETRY_HPP
