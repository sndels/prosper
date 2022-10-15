#ifndef PROSPER_VERTEX_HPP
#define PROSPER_VERTEX_HPP

#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos{0.f};
    glm::vec3 normal{0.f, 1.f, 0.f};
    glm::vec4 tangent{0.f, 0.f, 1.f, 0.f};
    glm::vec2 texCoord0{0.5f};
};
// Make Vertex packs tightly and the fields are also tight
static_assert(alignof(Vertex) == sizeof(float));
static_assert(offsetof(Vertex, pos) == 0);
static_assert(offsetof(Vertex, normal) == 3 * sizeof(float));
static_assert(offsetof(Vertex, tangent) == 6 * sizeof(float));
static_assert(offsetof(Vertex, texCoord0) == 10 * sizeof(float));

#endif // PROSPER_VERTEX_HPP
