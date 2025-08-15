#ifndef SHARED_SHADER_STRUCTS_SCENE_LIGHTS_H
#define SHARED_SHADER_STRUCTS_SCENE_LIGHTS_H

#include "../fields.h"

#ifdef __cplusplus

// Cant do {}s in macro arguments
#define DIRECTIONAL_LIGHT_DEFAULT_DIRECTION glm::vec4{-1.f, -1.f, -1.f, 1.f}

namespace scene::shader_structs
{

#endif // __cplusplus

struct DirectionalLightParameters
{
    STRUCT_FIELD_GLM(vec4, irradiance, 2.f);
    STRUCT_FIELD_GLM(vec4, direction, DIRECTIONAL_LIGHT_DEFAULT_DIRECTION);
};

struct PointLight
{
    STRUCT_FIELD_GLM(vec4, radianceAndRadius, 0.f);
    STRUCT_FIELD_GLM(vec4, position, 0.f);
};

struct SpotLight
{
    STRUCT_FIELD_GLM(vec4, radianceAndAngleScale, 0.f);
    STRUCT_FIELD_GLM(vec4, positionAndAngleOffset, 0.f);
    STRUCT_FIELD_GLM(vec4, direction, 0.f);
};

#ifdef __cplusplus
} //  namespace scene::shader_structs
#endif //  __cplusplus

#endif // SHARED_SHADER_STRUCTS_SCENE_LIGHTS_H
