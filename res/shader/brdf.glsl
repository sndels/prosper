#ifndef BRDF_GLSL
#define BRDF_GLSL

#include "common/math.glsl"
#include "scene/material.glsl"
#include "scene/visible_surface.glsl"

// Lambert diffuse term
vec3 lambertBRFD(vec3 c_diff) { return c_diff / PI; }

// Trowbridge-Reitz(GGX) distribution function
float trowbridgeReitz(float NoH, float alpha)
{
    // Match gltf spec
    float a2 = alpha * alpha;

    float denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

vec3 schlickFresnel(float VoH, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

// From https://seblagarde.wordpress.com/2011/08/17/
// via https://learnopengl.com/PBR/IBL/Diffuse-irradiance
vec3 schlickFresnelWithRoughness(float VoH, vec3 f0, float roughness)
{
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - VoH, 5.0);
}

// Schlick approximation of Smith's method for Trowbridge-Reitz(GGX)
// geometry function
float schlickTrowbridgeReitz(float NoL, float NoV, float alpha)
{
    float k = alpha * 0.5;
    // Avoid division by zero
    k = max(k, 0.0001);
    float gl = NoL / (NoL * (1.0 - k) + k);
    float gv = NoV / (NoV * (1.0 - k) + k);
    return gl * gv;
}

// Evaluate the Cook-Torrance specular BRDF
vec3 cookTorranceBRDF(
    float NoL, float NoV, float NoH, float VoH, vec3 f0, float roughness)
{
    // Match gltf spec
    float alpha = roughness * roughness;

    float D = trowbridgeReitz(NoH, alpha);
    vec3 F = schlickFresnel(VoH, f0);
    float G = schlickTrowbridgeReitz(NoL, NoV, alpha);

    float denom = 4.0 * NoL * NoV + 0.0001;
    return D * F * G / denom;
}

vec3 fresnelZero(VisibleSurface surface)
{
    return mix(
        vec3(0.04), surface.material.albedo.rgb, surface.material.metallic);
}

// Evaluate combined diffuse and specular BRDF
vec3 evalBRDFTimesNoL(vec3 l, VisibleSurface surface)
{
    // Common dot products
    vec3 h = normalize(surface.invViewRayWS + l);
    float NoL = saturate(dot(surface.normalWS, l));
    float NoH = saturate(dot(surface.normalWS, h));
    float VoH = saturate(dot(surface.invViewRayWS, h));

    // Use standard approximation of default fresnel
    vec3 f0 = fresnelZero(surface);

    // Match glTF spec
    vec3 c_diff =
        mix(surface.material.albedo.rgb * (1 - 0.04), vec3(0),
            surface.material.metallic);

    return (lambertBRFD(c_diff) +
            cookTorranceBRDF(
                NoL, surface.NoV, NoH, VoH, f0, surface.material.roughness)) *
           NoL;
}

#endif // BRDF_GLSL
