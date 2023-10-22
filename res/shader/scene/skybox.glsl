#ifndef SCENE_SKYBOX_GLSL
#define SCENE_SKYBOX_GLSL

layout(set = SKYBOX_SET, binding = 0) uniform samplerCube skybox;
layout(set = SKYBOX_SET, binding = 1) uniform samplerCube skyboxIrradiance;
layout(set = SKYBOX_SET, binding = 2) uniform sampler2D specularBrdfLut;
layout(set = SKYBOX_SET, binding = 3) uniform samplerCube skyboxRadiance;

#include "../brdf.glsl"

// Adapted from https://learnopengl.com/PBR/IBL/Diffuse-irradiance

vec3 evalIBL(VisibleSurface surface)
{
    vec3 f0 = fresnelZero(surface);

    vec3 F = schlickFresnelWithRoughness(
        max(dot(surface.normalWS, surface.invViewRayWS), 0.0), f0,
        surface.material.roughness);

    vec3 kD = 1.0 - F;
    kD *= 1 - surface.material.metallic;

    vec3 irradiance = texture(skyboxIrradiance, surface.normalWS).rgb;
    vec3 diffuse = irradiance * surface.material.albedo;

    // TODO: Apply AO to this
    vec3 ibl = kD * diffuse;

    return ibl;
}

#endif // SCENE_SKYBO_GLSL
