#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)

layout(set = 0, binding = 0) uniform Camera
{
    mat4 worldToCamera;
    mat4 cameraToClip;
    vec3 eye;
}
camera;

layout(set = 2, binding = 0) uniform sampler2D baseColor;
layout(set = 2, binding = 1) uniform sampler2D metallicRoughness;
layout(set = 2, binding = 2) uniform sampler2D tangentNormal;

layout(set = 3, binding = 0) uniform DirectionalLight
{
    vec4 irradiance;
    vec4 direction;
}
directionalLight;

// This needs to match the engine
#define MAX_POINT_LIGHT_COUNT 100000

struct PointLight
{
    vec4 radiance;
    vec4 position;
};

layout(set = 3, binding = 1) buffer PointLights
{
    PointLight lights[MAX_POINT_LIGHT_COUNT];
    uint count;
}
pointLights;

// This needs to match the engine
#define MAX_SPOT_LIGHT_COUNT 100000

struct SpotLight
{
    vec4 radianceAndAngleScale;
    vec4 positionAndAngleOffset;
    vec4 direction;
};

layout(set = 3, binding = 2) buffer SpotLights
{
    SpotLight lights[MAX_SPOT_LIGHT_COUNT];
    uint count;
}
spotLights;

// Needs to match Material::PCBlock
layout(push_constant) uniform MaterialPC
{
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaMode;
    float alphaCutoff;
    int baseColorTextureSet;
    int metallicRoughnessTextureSet;
    int normalTextureSet;
}
materialPC;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec2 fragTexCoord0;
layout(location = 2) in mat3 fragTBN;

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
    vec4 linearBaseColor = sRGBtoLinear(texture(baseColor, fragTexCoord0)) *
                           materialPC.baseColorFactor;

    // Alpha masking is 1.f
    if (materialPC.alphaMode > 0.f && materialPC.alphaMode < 2.f)
    {
        if (linearBaseColor.a < materialPC.alphaCutoff)
            discard;
    }

    float metallic;
    float roughness;
    if (materialPC.metallicRoughnessTextureSet > -1)
    {
        vec3 mr = texture(metallicRoughness, fragTexCoord0).rgb;
        metallic = mr.b * materialPC.metallicFactor;
        roughness = mr.g * materialPC.roughnessFactor;
    }
    else
    {
        metallic = materialPC.metallicFactor;
        roughness = materialPC.roughnessFactor;
    }

    vec3 normal;
    if (materialPC.normalTextureSet > -1)
    {
        mat3 TBN = length(fragTBN[0]) > 0 ? fragTBN : generateTBN();
        normal = normalize(
            TBN * (texture(tangentNormal, fragTexCoord0).xyz * 2 - 1));
    }
    else
        normal = normalize(fragTBN[2]);

    Material m;
    m.albedo = linearBaseColor.rgb;
    m.metallic = metallic;
    m.roughness = roughness;

    vec3 v = normalize(camera.eye - fragPosition);

    vec3 color = vec3(0);
    {
        vec3 l = -normalize(directionalLight.direction.xyz);
        color += directionalLight.irradiance.xyz * evalBRDF(normal, v, l, m);
    }

    for (int i = 0; i < pointLights.count; ++i)
    {
        PointLight light = pointLights.lights[i];
        vec3 toLight = light.position.xyz - fragPosition;
        float d2 = dot(toLight, toLight);
        vec3 l = toLight / sqrt(d2);
        color += light.radiance.xyz * evalBRDF(normal, v, l, m) / d2;
    }

    for (int i = 0; i < spotLights.count; ++i)
    {
        SpotLight light = spotLights.lights[i];
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

    // Alpha blending is 2.f
    float alpha = materialPC.alphaMode > 1.f ? linearBaseColor.a : 1.f;

    outColor = vec4(color, alpha);
}
