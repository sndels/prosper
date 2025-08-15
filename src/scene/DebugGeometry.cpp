#include "DebugGeometry.hpp"

#include "utils/Utils.hpp"

namespace scene
{

void DebugLines::reset() { count = 0; }

void DebugLines::addLine(glm::vec3 p0, glm::vec3 p1, glm::vec3 color)
{
    glm::vec3 *lineData = static_cast<glm::vec3 *>(buffer.mapped) +
                          (asserted_cast<size_t>(count) * 3);
    lineData[0] = p0;
    lineData[1] = p1;
    lineData[2] = color;
    count++;
}

} // namespace scene
