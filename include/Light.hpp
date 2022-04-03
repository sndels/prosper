#ifndef PROSPER_LIGHT_HPP
#define PROSPER_LIGHT_HPP

#include "Device.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <vector>

struct DirectionalLight
{
    struct Parameters
    {
        // Use vec4 because vec3 alignment is no fun between glsl, c++
        glm::vec4 irradiance{2.f};
        glm::vec4 direction{-1.f, -1.f, -1.f, 1.f};
    } parameters;

    std::vector<Buffer> uniformBuffers;

    [[nodiscard]] std::vector<vk::DescriptorBufferInfo> bufferInfos() const;

    void updateBuffer(uint32_t nextImage) const;
};

struct PointLights
{
    static const uint32_t max_count = 1024;
    struct PointLight
    {
        glm::vec4 radianceAndRadius{0.f};
        glm::vec4 position{0.f};
    };

    struct BufferData
    {
        std::array<PointLight, max_count> lights;
        uint32_t count{0};
    } bufferData;

    std::vector<Buffer> storageBuffers;

    [[nodiscard]] std::vector<vk::DescriptorBufferInfo> bufferInfos() const;

    void updateBuffer(uint32_t nextImage) const;
};

struct SpotLights
{
    static const uint32_t max_count = 1024;
    struct SpotLight
    {
        glm::vec4 radianceAndAngleScale{0.f};
        glm::vec4 positionAndAngleOffset{0.f};
        glm::vec4 direction{0.f};
    };

    struct BufferData
    {
        std::array<SpotLight, max_count> lights;
        uint32_t count{0};
    } bufferData;

    std::vector<Buffer> storageBuffers;

    [[nodiscard]] std::vector<vk::DescriptorBufferInfo> bufferInfos() const;

    void updateBuffer(uint32_t nextImage) const;
};

#endif // PROSPER_LIGHT_HPP
