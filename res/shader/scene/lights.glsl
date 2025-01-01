#ifndef SCENE_LIGHTS_GLSL
#define SCENE_LIGHTS_GLSL

#include "../shared/shader_structs/scene/lights.h"

layout(set = LIGHTS_SET, binding = 0) readonly buffer DirectionalLightDSB
{
    DirectionalLightParameters directionalLight;
};

layout(set = LIGHTS_SET, binding = 1) readonly buffer PointLightsDSB
{
    PointLight lights[MAX_POINT_LIGHT_COUNT];
    uint count;
}
pointLights;

layout(set = LIGHTS_SET, binding = 2) readonly buffer SpotLightsDSB
{
    SpotLight lights[MAX_SPOT_LIGHT_COUNT];
    uint count;
}
spotLights;

#endif // SCENE_LIGHTS_GLSL
