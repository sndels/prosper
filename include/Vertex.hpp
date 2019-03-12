#ifndef PROSPER_VERTEX_HPP
#define PROSPER_VERTEX_HPP

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription bindingDescription()
    {
        VkVertexInputBindingDescription description = {};
        description.binding = 0;
        description.stride = sizeof(Vertex);
        description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return description;
    }

    static std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> descriptions = {};

        // pos
        descriptions[0].binding = 0;
        descriptions[0].location = 0;
        descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        descriptions[0].offset = offsetof(Vertex, pos);

        // color
        descriptions[1].binding = 0;
        descriptions[1].location = 1;
        descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        descriptions[1].offset = offsetof(Vertex, color);

        return descriptions;
    }
};

#endif // PROSPER_VERTEX_HPP
