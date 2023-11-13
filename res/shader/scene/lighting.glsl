
#ifndef SCENE_LIGHTING_GLSL
#define SCENE_LIGHTING_GLSL

#include "../brdf.glsl"
#include "lights.glsl"
#include "visible_surface.glsl"

vec3 evalDirectionalLight(VisibleSurface surface)
{
    vec3 l = -normalize(directionalLight.direction.xyz);
    return directionalLight.irradiance.xyz * evalBRDFTimesNoL(l, surface);
}

void evaluateUnshadowedPointLight(
    VisibleSurface surface, uint pointIndex, out vec3 l, inout float d,
    out vec3 irradiance)
{
    PointLight light = pointLights.lights[pointIndex];
    vec3 pos = light.position.xyz;

    vec3 radiance = light.radianceAndRadius.xyz;
    float radius = light.radianceAndRadius.w;

    vec3 toLight = pos - surface.positionWS;
    float d2 = dot(toLight, toLight);
    d = sqrt(d2);

    l = toLight / d;

    float dPerR = d / radius;
    float dPerR2 = dPerR * dPerR;
    float dPerR4 = dPerR2 * dPerR2;
    float radialAttenuation = max(min(1.0 - dPerR4, 1), 0);

    irradiance = radiance * radialAttenuation / d2;
}

void evaluateUnshadowedSpotLight(
    VisibleSurface surface, uint spotIndex, inout vec3 l, inout float d,
    out vec3 irradiance)
{
    SpotLight light = spotLights.lights[spotIndex];
    vec3 toLight = light.positionAndAngleOffset.xyz - surface.positionWS;
    float d2 = dot(toLight, toLight);
    d = sqrt(d2);
    l = toLight / d;

    // Angular attenuation from gltf spec
    float cd = dot(-light.direction.xyz, l);
    float angularAttenuation = saturate(
        cd * light.radianceAndAngleScale.w + light.positionAndAngleOffset.w);
    angularAttenuation *= angularAttenuation;

    irradiance = angularAttenuation * light.radianceAndAngleScale.xyz / d2;
}

#endif // SCENE_LIGHTING_GLSL
