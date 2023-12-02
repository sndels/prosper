#ifndef PROSPER_SCENE_LIGHT_HPP
#define PROSPER_SCENE_LIGHT_HPP

#include "../gfx/Fwd.hpp"
#include "../utils/Utils.hpp"

#include <glm/glm.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

struct DirectionalLight
{
    struct Parameters
    {
        // Use vec4 because vec3 alignment is no fun between glsl, c++
        glm::vec4 irradiance{2.f};
        glm::vec4 direction{-1.f, -1.f, -1.f, 1.f};
    } parameters;

    static const uint32_t sBufferByteSize = sizeof(Parameters);

    [[nodiscard]] uint32_t write(RingBuffer &buffer) const;
};

struct PointLight
{
    glm::vec4 radianceAndRadius{0.f};
    glm::vec4 position{0.f};
};

struct PointLights
{
    static const uint32_t sMaxCount = 1024;

    static void appendShaderDefines(wheels::String &str)
    {
        appendDefineStr(str, "MAX_POINT_LIGHT_COUNT", PointLights::sMaxCount);
    };

    wheels::StaticArray<PointLight, sMaxCount> data;

    // Light data and uint32_t count
    static const uint32_t sBufferByteSize =
        sMaxCount * sizeof(PointLight) + sizeof(uint32_t);

    [[nodiscard]] uint32_t write(RingBuffer &buffer) const;
};

struct SpotLight
{
    glm::vec4 radianceAndAngleScale{0.f};
    glm::vec4 positionAndAngleOffset{0.f};
    glm::vec4 direction{0.f};
};

struct SpotLights
{
    static const uint32_t sMaxCount = 1024;

    static void appendShaderDefines(wheels::String &str)
    {
        appendDefineStr(str, "MAX_SPOT_LIGHT_COUNT", SpotLights::sMaxCount);
    }

    wheels::StaticArray<SpotLight, sMaxCount> data;

    // Light data and uint32_t count
    static const uint32_t sBufferByteSize =
        sMaxCount * sizeof(SpotLight) + sizeof(uint32_t);

    [[nodiscard]] uint32_t write(RingBuffer &buffer) const;
};

#endif // PROSPER_SCENE_LIGHT_HPP
