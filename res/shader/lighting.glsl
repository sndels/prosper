
#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

#include "lights.glsl"
#include "visible_surface.glsl"

vec3 evalDirectionalLight(VisibleSurface surface)
{
    vec3 l = -normalize(directionalLight.direction.xyz);
    return directionalLight.irradiance.xyz * evalBRDF(l, surface);
}

vec3 evalPointLights(VisibleSurface surface, LightClusterInfo lightInfo)
{
    vec3 color = vec3(0);
    for (uint i = 0; i < lightInfo.pointCount; ++i)
    {
        uint index = imageLoad(lightIndices, int(lightInfo.indexOffset + i)).x;
        PointLight light = pointLights.lights[index];
        vec3 pos = light.position.xyz;
        vec3 radiance = light.radianceAndRadius.xyz;
        float radius = light.radianceAndRadius.w;

        vec3 toLight = pos - surface.positionWS;
        float d2 = dot(toLight, toLight);
        float d = sqrt(d2);

        vec3 l = toLight / d;

        float dPerR = d / radius;
        float dPerR2 = dPerR * dPerR;
        float dPerR4 = dPerR2 * dPerR2;
        float attenuation = max(min(1.0 - dPerR4, 1), 0) / d2;

        color += radiance * attenuation * evalBRDF(l, surface);
    }
    return color;
}

vec3 evalSpotLights(VisibleSurface surface, LightClusterInfo lightInfo)
{
    vec3 color = vec3(0);
    for (uint i = 0; i < lightInfo.spotCount; ++i)
    {
        uint index = imageLoad(
                         lightIndices,
                         int(lightInfo.indexOffset + lightInfo.pointCount + i))
                         .x;
        SpotLight light = spotLights.lights[index];
        vec3 toLight = light.positionAndAngleOffset.xyz - surface.positionWS;
        float d2 = dot(toLight, toLight);
        vec3 l = toLight / sqrt(d2);

        // Angular attenuation rom gltf spec
        float cd = dot(-light.direction.xyz, l);
        float angularAttenuation = saturate(
            cd * light.radianceAndAngleScale.w +
            light.positionAndAngleOffset.w);
        angularAttenuation *= angularAttenuation;

        color += angularAttenuation * light.radianceAndAngleScale.xyz *
                 evalBRDF(l, surface) / d2;
    }
    return color;
}

#endif // LIGHTING_GLSL
