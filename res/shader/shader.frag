#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)

layout(set = 0, binding = 0) uniform Camera {
    mat4 worldToCamera;
    mat4 cameraToClip;
    vec3 eye;
} camera;

layout(set = 2, binding = 0) uniform sampler2D baseColor;
layout(set = 2, binding = 1) uniform sampler2D metallicRoughness;
layout(set = 2, binding = 2) uniform sampler2D normal;

// Needs to match Material::PCBlock
layout(push_constant) uniform MaterialPC {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureSet;
    int metallicRoughnessTextureSet;
    int normalTextureSet;
} materialPC;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec2 fragTexCoord0;
layout(location = 2) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

// TODO: Uniform inputs
vec3 light_dir = vec3(-1, -1, -1);
vec3 light_int = vec3(4);

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
};

// TODO: sRGB-texture for baseColor
float sRGBtoLinear(float x)
{
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}
vec3 sRGBtoLinear(vec3 v)
{
    return vec3(sRGBtoLinear(v.r), sRGBtoLinear(v.g), sRGBtoLinear(v.b));
}

// Lambert diffuse term
vec3 lambertBRFD(vec3 c_diff)
{
    return c_diff / PI;
}

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
vec3 cookTorranceBRDF(float NoL, float NoV, float NoH, float VoH, vec3 f0, float roughness)
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

    return (lambertBRFD(c_diff) + cookTorranceBRDF(NoL, NoV, NoH, VoH, f0, m.roughness)) * NoL;
}

void main()
{
    // TODO: alpha, skipping normal mapping based on B
    vec3 albedo = sRGBtoLinear(texture(baseColor, fragTexCoord0).rgb);
    vec3 mr = texture(metallicRoughness, fragTexCoord0).rgb;
    vec3 normal = normalize(fragTBN * (texture(normal, fragTexCoord0).rgb * 2 - 1));
    Material m;
    m.albedo = albedo * materialPC.baseColorFactor.rgb;
    m.metallic = mr.b * materialPC.metallicFactor;
    m.roughness = mr.g * materialPC.roughnessFactor;

    vec3 v = normalize(camera.eye - fragPosition);
    vec3 l = -normalize(light_dir);
    vec3 color = light_int * evalBRDF(normal, v, l, m);
    // TODO: Doesn't outputting to sRGB render target do this?
    float gamma = 2.2;
    outColor = vec4(pow(color, vec3(1 / gamma)), 1.f);
}
