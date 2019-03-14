#ifndef PROSPER_VERTEX_HPP
#define PROSPER_VERTEX_HPP

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static const vk::VertexInputBindingDescription& bindingDescription()
    {
        static const vk::VertexInputBindingDescription description(
            0,
            sizeof(Vertex),
            vk::VertexInputRate::eVertex
        );

        return description;
    }

    static const std::array<vk::VertexInputAttributeDescription, 2>& attributeDescriptions()
    {
        static const std::array<vk::VertexInputAttributeDescription, 2> descriptions{
            // pos
            vk::VertexInputAttributeDescription(
                0, // location
                0, // binding
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, pos)
            ),
            // color
            vk::VertexInputAttributeDescription(
                1, // location
                0, // buinding
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, color)
            )
        };

        return descriptions;
    }
};

#endif // PROSPER_VERTEX_HPP
