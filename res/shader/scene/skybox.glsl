#ifndef SCENE_SKYBOX_GLSL
#define SCENE_SKYBOX_GLSL

layout(set = SKYBOX_SET, binding = 0) uniform samplerCube skybox;
layout(set = SKYBOX_SET, binding = 1) uniform samplerCube skyboxIrradiance;
layout(set = SKYBOX_SET, binding = 2) uniform sampler2D specularBrdfLut;
layout(set = SKYBOX_SET, binding = 3) uniform samplerCube skyboxRadiance;

#include "../brdf.glsl"

// Adapted from
// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
// https://learnopengl.com/PBR/IBL/Specular-IBL

vec3 evalIBL(VisibleSurface surface)
{
    vec3 f0 = fresnelZero(surface);

    float NoV = saturate(dot(surface.normalWS, surface.invViewRayWS));

    vec3 F = schlickFresnelWithRoughness(NoV, f0, surface.material.roughness);

    vec3 kD = 1.0 - F;
    kD *= 1 - surface.material.metallic;

    // Diffuse
    vec3 irradiance = texture(skyboxIrradiance, surface.normalWS).rgb;
    vec3 diffuse = irradiance * surface.material.albedo;

    // Specular
    vec3 R = reflect(-surface.invViewRayWS, surface.normalWS);
    // TODO: This should be a define
    const float MAX_REFLECTION_LOD = 10.0;
    vec3 prefilteredRadiance =
        textureLod(
            skyboxRadiance, R, surface.material.roughness * MAX_REFLECTION_LOD)
            .rgb;
    vec2 envBrdf =
        texture(specularBrdfLut, vec2(NoV, 1 - surface.material.roughness)).rg;
    vec3 specular = prefilteredRadiance * (F * envBrdf.x + envBrdf.y);

    // TODO: Apply AO to this
    vec3 ibl = kD * diffuse + specular;

    return ibl;
}

#endif // SCENE_SKYBO_GLSL
