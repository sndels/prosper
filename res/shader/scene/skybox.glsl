#ifndef SCENE_SKYBOX_GLSL
#define SCENE_SKYBOX_GLSL

layout(set = SKYBOX_SET, binding = 0) uniform samplerCube skybox;
layout(set = SKYBOX_SET, binding = 1) uniform samplerCube skyboxIrradiance;
layout(set = SKYBOX_SET, binding = 2) uniform sampler2D specularBrdfLut;
layout(set = SKYBOX_SET, binding = 3) uniform samplerCube skyboxRadiance;

#include "../brdf.glsl"
#include "../common/random.glsl"
#include "../common/sampling.glsl"

// From Real Shading in Unreal Engine 4
// by Brian Karis
vec3 specularIBLReference(vec3 SpecularColor, float Roughness, vec3 N, vec3 V)
{
    float alpha = Roughness * Roughness;
    vec3 SpecularLighting = vec3(0);
    const uint NumSamples = 1024;
    for (uint i = 0; i < NumSamples; i++)
    {
        vec2 Xi = hammersley(i, NumSamples);
        vec3 H = importanceSampleIBLTrowbridgeReitz(Xi, alpha, N);
        vec3 L = 2 * dot(V, H) * H - V;
        float NoV = saturate(dot(N, V));
        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        float VoH = saturate(dot(V, H));
        if (NoL > 0)
        {
            vec3 SampleColor = min(textureLod(skybox, L, 0).rgb, 10);
            float G = schlickTrowbridgeReitz(NoL, NoV, alpha);
            float Fc = pow(1 - VoH, 5);
            vec3 F = (1 - Fc) * SpecularColor + Fc;
            // Incident light = SampleColor * NoL
            // Microfacet specular = D*G*F / (4*NoL*NoV)
            // pdf = D * NoH / (4 * VoH)
            SpecularLighting += SampleColor * F * G * VoH / (NoH * NoV);
        }
    }
    return SpecularLighting / NumSamples;
}

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
        texture(specularBrdfLut, vec2(NoV, surface.material.roughness)).rg;

    // vec3 specular = specularIBLReference(
    //     f0, surface.material.roughness, surface.normalWS,
    //     surface.invViewRayWS);
    vec3 specular = prefilteredRadiance * (F * envBrdf.x + envBrdf.y);

    // TODO: Apply AO to this
    vec3 ibl = kD * diffuse + specular;

    return ibl;
}

#endif // SCENE_SKYBO_GLSL
