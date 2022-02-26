#ifndef PROSPER_VERTEX_HPP
#define PROSPER_VERTEX_HPP

#include "vulkan.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <array>

struct Vertex
{
    using AttributeDescriptions =
        std::array<vk::VertexInputAttributeDescription, 4>;

    glm::vec3 pos = glm::vec3{0.f};
    glm::vec3 normal = glm::vec3{0.f, 1.f, 0.f};
    glm::vec4 tangent = glm::vec4{0.f, 0.f, 1.f, 0.f};
    glm::vec2 texCoord0 = glm::vec2{0.5f};

    static const vk::VertexInputBindingDescription &bindingDescription()
    {
        static const vk::VertexInputBindingDescription description{
            0, sizeof(Vertex), vk::VertexInputRate::eVertex};

        return description;
    }

    static const AttributeDescriptions &attributeDescriptions()
    {
        static const std::array<vk::VertexInputAttributeDescription, 4>
            descriptions{
                vk::VertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = vk::Format::eR32G32B32Sfloat,
                    .offset = offsetof(Vertex, pos)},
                vk::VertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = vk::Format::eR32G32B32Sfloat,
                    .offset = offsetof(Vertex, normal)},
                vk::VertexInputAttributeDescription{
                    .location = 2,
                    .binding = 0,
                    .format = vk::Format::eR32G32B32A32Sfloat,
                    .offset = offsetof(Vertex, tangent)},
                vk::VertexInputAttributeDescription{
                    .location = 3,
                    .binding = 0,
                    .format = vk::Format::eR32G32Sfloat,
                    .offset = offsetof(Vertex, texCoord0)}};

        return descriptions;
    }
};

#endif // PROSPER_VERTEX_HPP
