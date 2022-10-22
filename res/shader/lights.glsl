#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

layout(set = LIGHTS_SET, binding = 0) uniform DirectionalLight
{
    vec4 irradiance;
    vec4 direction;
}
directionalLight;

struct PointLight
{
    vec4 radianceAndRadius;
    vec4 position;
};

layout(set = LIGHTS_SET, binding = 1) buffer PointLights
{
    PointLight lights[MAX_POINT_LIGHT_COUNT];
    uint count;
}
pointLights;

struct SpotLight
{
    vec4 radianceAndAngleScale;
    vec4 positionAndAngleOffset;
    vec4 direction;
};

layout(set = LIGHTS_SET, binding = 2) buffer SpotLights
{
    SpotLight lights[MAX_SPOT_LIGHT_COUNT];
    uint count;
}
spotLights;

#endif // LIGHTS_GLSL
