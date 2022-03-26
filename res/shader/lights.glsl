layout(set = LIGHTS_SET, binding = 0) uniform DirectionalLight
{
    vec4 irradiance;
    vec4 direction;
}
directionalLight;

// This needs to match the engine
#define MAX_POINT_LIGHT_COUNT 1024

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

// This needs to match the engine
#define MAX_SPOT_LIGHT_COUNT 1024

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
