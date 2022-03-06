#version 450
#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)

#include "ds0_lights.glsl"

#include "ds1_camera.glsl"

#include "ds2_light_clusters.glsl"

const uint AlphaModeOpaque = 0;
const uint AlphaModeMask = 1;
const uint AlphaModeBlend = 2;

struct MaterialData
{
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    uint alphaMode;
    uint baseColorTexture;
    uint metallicRoughnessTexture;
    uint normalTexture;
    uint pad;
};

layout(std430, set = 3, binding = 0) readonly buffer MaterialDatas
{
    MaterialData materials[];
}
materialDatas;
layout(set = 3, binding = 1) uniform sampler2D materialTextures[];

#include "pc_mesh.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in float fragZCam;
layout(location = 2) in vec2 fragTexCoord0;
layout(location = 3) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

struct Material
{
    vec3 albedo;
    float metallic;
    float roughness;
};

float sRGBtoLinear(float x)
{
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}
vec3 sRGBtoLinear(vec3 v)
{
    return vec3(sRGBtoLinear(v.r), sRGBtoLinear(v.g), sRGBtoLinear(v.b));
}
// Alpha shouldn't be converted
vec4 sRGBtoLinear(vec4 v) { return vec4(sRGBtoLinear(v.rgb), v.a); }

mat3 generateTBN()
{
    // http://www.thetenthplanet.de/archives/1180
    vec3 dp1 = dFdx(fragPosition);
    vec3 dp2 = dFdy(fragPosition);
    vec2 duv1 = dFdx(fragTexCoord0);
    vec2 duv2 = dFdy(fragTexCoord0);

    vec3 N = normalize(fragTBN[2]);
    vec3 T = normalize(dp1 * duv2.t - dp2 * duv1.t);
    vec3 B = normalize(cross(N, T));
    return mat3(T, B, N);
}

// Lambert diffuse term
vec3 lambertBRFD(vec3 c_diff) { return c_diff / PI; }

// GGX distribution function
float ggx(float NoH, float alpha)
{
    // Match gltf spec
    float a2 = alpha * alpha;

    float denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick fresnel function
vec3 schlickFresnel(float VoH, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

// Schlick-GGX geometry function
float schlick_ggx(float NoL, float NoV, float alpha)
{
    float k = alpha + 1.0;
    k *= k * 0.125;
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

    float D = ggx(NoH, alpha);
    vec3 F = schlickFresnel(VoH, f0);
    float G = schlick_ggx(NoL, NoV, alpha);

    float denom = 4.0 * NoL * NoV + 0.0001;
    return D * F * G / denom;
}

// Evaluate combined diffuse and specular BRDF
vec3 evalBRDF(vec3 n, vec3 v, vec3 l, Material m)
{
    // Common dot products
    vec3 h = normalize(v + l);
    float NoV = saturate(dot(n, v));
    float NoL = saturate(dot(n, l));
    float NoH = saturate(dot(n, h));
    float VoH = saturate(dot(v, h));

    // Use standard approximation of default fresnel
    vec3 f0 = mix(vec3(0.04), m.albedo, m.metallic);

    // Match glTF spec
    vec3 c_diff = mix(m.albedo * (1 - 0.04), vec3(0), m.metallic);

    return (lambertBRFD(c_diff) +
            cookTorranceBRDF(NoL, NoV, NoH, VoH, f0, m.roughness)) *
           NoL;
}

void main()
{
    MaterialData material = materialDatas.materials[meshPC.MaterialID];

    vec4 linearBaseColor;
    uint baseColorTex = material.baseColorTexture;
    if (baseColorTex > 0)
        linearBaseColor = sRGBtoLinear(
            texture(materialTextures[baseColorTex], fragTexCoord0));
    else
        linearBaseColor = vec4(1);
    linearBaseColor *= material.baseColorFactor;

    if (material.alphaMode == AlphaModeMask)
    {
        if (linearBaseColor.a < material.alphaCutoff)
            discard;
    }

    float metallic;
    float roughness;
    uint metallicRoughnessTex = material.metallicRoughnessTexture;
    if (metallicRoughnessTex > 0)
    {
        vec3 mr =
            texture(materialTextures[metallicRoughnessTex], fragTexCoord0).rgb;
        metallic = mr.b * material.metallicFactor;
        roughness = mr.g * material.roughnessFactor;
    }
    else
    {
        metallic = material.metallicFactor;
        roughness = material.roughnessFactor;
    }

    vec3 normal;
    uint normalTextureTex = material.normalTexture;
    if (normalTextureTex > 0)
    {
        mat3 TBN = length(fragTBN[0]) > 0 ? fragTBN : generateTBN();
        normal = normalize(
            TBN *
            (texture(materialTextures[normalTextureTex], fragTexCoord0).xyz *
                 2 -
             1));
    }
    else
        normal = normalize(fragTBN[2]);

    Material m;
    m.albedo = linearBaseColor.rgb;
    m.metallic = metallic;
    m.roughness = roughness;

    vec3 v = normalize(camera.eye.xyz - fragPosition);

    vec3 color = vec3(0);
    {
        vec3 l = -normalize(directionalLight.direction.xyz);
        color += directionalLight.irradiance.xyz * evalBRDF(normal, v, l, m);
    }

    uvec3 ci = clusterIndex(uvec2(gl_FragCoord.xy), fragZCam);

    uint clusterIndexOffset, pointCount, spotCount;
    unpackClusterPointer(ci, clusterIndexOffset, pointCount, spotCount);

    for (uint i = 0; i < pointCount; ++i)
    {
        uint index = imageLoad(lightIndices, int(clusterIndexOffset + i)).x;
        PointLight light = pointLights.lights[index];
        vec3 pos = light.position.xyz;
        vec3 radiance = light.radianceAndRadius.xyz;
        float radius = light.radianceAndRadius.w;

        vec3 toLight = pos - fragPosition;
        float d2 = dot(toLight, toLight);
        float d = sqrt(d2);

        vec3 l = toLight / d;

        float dPerR = d / radius;
        float dPerR2 = dPerR * dPerR;
        float dPerR4 = dPerR2 * dPerR2;
        float attenuation = max(min(1.0 - dPerR4, 1), 0) / d2;

        color += radiance * attenuation * evalBRDF(normal, v, l, m);
    }

    for (uint i = 0; i < spotCount; ++i)
    {
        uint index =
            imageLoad(lightIndices, int(clusterIndexOffset + pointCount + i)).x;
        SpotLight light = spotLights.lights[index];
        vec3 toLight = light.positionAndAngleOffset.xyz - fragPosition;
        float d2 = dot(toLight, toLight);
        vec3 l = toLight / sqrt(d2);

        // Angular attenuation rom gltf spec
        float cd = dot(-light.direction.xyz, l);
        float angularAttenuation = saturate(
            cd * light.radianceAndAngleScale.w +
            light.positionAndAngleOffset.w);
        angularAttenuation *= angularAttenuation;

        color += angularAttenuation * light.radianceAndAngleScale.xyz *
                 evalBRDF(normal, v, l, m) / d2;
    }

    float alpha =
        material.alphaMode == AlphaModeBlend ? linearBaseColor.a : 1.f;

    outColor = vec4(color, alpha);
}
