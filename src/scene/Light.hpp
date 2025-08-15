#ifndef PROSPER_SCENE_LIGHT_HPP
#define PROSPER_SCENE_LIGHT_HPP

#include "gfx/Fwd.hpp"
#include "utils/Utils.hpp"

#include <glm/glm.hpp>
#include <shader_structs/scene/lights.h>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/span.hpp>

namespace scene
{

struct DirectionalLight
{
    shader_structs::DirectionalLightParameters parameters;

    static const uint32_t sBufferByteSize =
        sizeof(shader_structs::DirectionalLightParameters);

    [[nodiscard]] uint32_t write(gfx::RingBuffer &buffer) const;
};

struct PointLights
{
    static const uint32_t sMaxCount = 1024;

    static void appendShaderDefines(wheels::String &str)
    {
        appendDefineStr(str, "MAX_POINT_LIGHT_COUNT", PointLights::sMaxCount);
    };

    wheels::InlineArray<shader_structs::PointLight, sMaxCount> data;

    // Light data and uint32_t count
    static const uint32_t sBufferByteSize =
        (sMaxCount * sizeof(shader_structs::PointLight)) + sizeof(uint32_t);

    [[nodiscard]] uint32_t write(gfx::RingBuffer &buffer) const;
};

struct SpotLights
{
    static const uint32_t sMaxCount = 1024;

    static void appendShaderDefines(wheels::String &str)
    {
        appendDefineStr(str, "MAX_SPOT_LIGHT_COUNT", SpotLights::sMaxCount);
    }

    wheels::InlineArray<shader_structs::SpotLight, sMaxCount> data;

    // Light data and uint32_t count
    static const uint32_t sBufferByteSize =
        (sMaxCount * sizeof(shader_structs::SpotLight)) + sizeof(uint32_t);

    [[nodiscard]] uint32_t write(gfx::RingBuffer &buffer) const;
};

} // namespace scene

#endif // PROSPER_SCENE_LIGHT_HPP
