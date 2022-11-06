#include "DebugGeometry.hpp"

void DebugLines::reset() { count = 0; }

void DebugLines::addLine(glm::vec3 p0, glm::vec3 p1, glm::vec3 color)
{
    reinterpret_cast<glm::vec3 *>(buffer.mapped)[count * 3] = p0;
    reinterpret_cast<glm::vec3 *>(buffer.mapped)[count * 3 + 1] = p1;
    reinterpret_cast<glm::vec3 *>(buffer.mapped)[count * 3 + 2] = color;
    count++;
}
