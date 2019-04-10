#ifndef PROSPER_VERTEX_HPP
#define PROSPER_VERTEX_HPP

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord0;

    static const vk::VertexInputBindingDescription& bindingDescription()
    {
        static const vk::VertexInputBindingDescription description{
            0,
            sizeof(Vertex),
            vk::VertexInputRate::eVertex
        };

        return description;
    }

    static const std::array<vk::VertexInputAttributeDescription, 3>& attributeDescriptions()
    {
        static const std::array<vk::VertexInputAttributeDescription, 3> descriptions{
            // pos
            vk::VertexInputAttributeDescription{
                0, // location
                0, // binding
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, pos)
            },
            // normal
            vk::VertexInputAttributeDescription{
                1, // location
                0, // binding
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, normal)
            },
            // texCoord0
            vk::VertexInputAttributeDescription{
                2, // location
                0, // binding
                vk::Format::eR32G32Sfloat,
                offsetof(Vertex, texCoord0)
            }
        };

        return descriptions;
    }
};

#endif // PROSPER_VERTEX_HPP
