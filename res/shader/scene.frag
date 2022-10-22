#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)

#include "camera.glsl"
#include "debug.glsl"
#include "light_clusters.glsl"
#include "lights.glsl"
#include "materials.glsl"
#include "random.glsl"
#include "scene_pc.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in float fragZCam;
layout(location = 2) in vec2 fragTexCoord0;
layout(location = 3) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

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
    vec3 f0 = mix(vec3(0.04), m.albedo.rgb, m.metallic);

    // Match glTF spec
    vec3 c_diff = mix(m.albedo.rgb * (1 - 0.04), vec3(0), m.metallic);

    return (lambertBRFD(c_diff) +
            cookTorranceBRDF(NoL, NoV, NoH, VoH, f0, m.roughness)) *
           NoL;
}

void main()
{
    Material material = sampleMaterial(scenePC.MaterialID, fragTexCoord0);

    // Early out if alpha test failed / zero alpha
    if (material.alpha == 0)
        discard;

    vec3 normal;
    if (material.normal.x != -2) // -2 signals no material normal
    {
        mat3 TBN = length(fragTBN[0]) > 0 ? fragTBN : generateTBN();
        normal = normalize(TBN * material.normal.xyz);
    }
    else
        normal = normalize(fragTBN[2]);

    vec3 v = normalize(camera.eye.xyz - fragPosition);

    vec3 color = vec3(0);
    {
        vec3 l = -normalize(directionalLight.direction.xyz);
        color +=
            directionalLight.irradiance.xyz * evalBRDF(normal, v, l, material);
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

        color += radiance * attenuation * evalBRDF(normal, v, l, material);
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
                 evalBRDF(normal, v, l, material) / d2;
    }

    float alpha = material.alpha > 0 ? material.alpha : 1.0;

    if (scenePC.DrawType >= CommonDebugDrawTypeOffset)
    {
        DebugInputs di;
        di.meshID = scenePC.MeshID;
        di.primitiveID = gl_PrimitiveID;
        di.materialID = scenePC.MaterialID;
        di.normal = normal;
        uint drawType = scenePC.DrawType - CommonDebugDrawTypeOffset;
        outColor = vec4(commonDebugDraw(drawType, di, material), 1);
        return;
    }

    outColor = vec4(color, alpha);
}
