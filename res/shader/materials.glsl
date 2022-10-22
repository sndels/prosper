#ifndef MATERIALS_GLSL
#define MATERIALS_GLSL

#include "material.glsl"

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

layout(std430, set = MATERIALS_SET, binding = 0) readonly buffer MaterialDatas
{
    MaterialData materials[];
}
materialDatas;
layout(set = MATERIALS_SET, binding = 1) uniform sampler
    materialSamplers[NUM_MATERIAL_SAMPLERS];
layout(set = MATERIALS_SET, binding = 1 + NUM_MATERIAL_SAMPLERS)
    uniform texture2D materialTextures[];

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

Material sampleMaterial(uint index, vec2 uv)
{
    MaterialData data = materialDatas.materials[index];
    Material ret;

    vec4 linearBaseColor;
    uint baseColorTex = data.baseColorTexture & 0xFFFFFF;
    uint baseColorSampler = data.baseColorTexture >> 24;
    if (baseColorTex > 0)
        linearBaseColor = sRGBtoLinear(texture(
            sampler2D(
                materialTextures[baseColorTex],
                materialSamplers[baseColorSampler]),
            uv));
    else
        linearBaseColor = vec4(1);
    linearBaseColor *= data.baseColorFactor;

    if (data.alphaMode == AlphaModeBlend)
        ret.alpha = linearBaseColor.a;
    else
    {
        if (data.alphaMode == AlphaModeMask)
        {
            if (linearBaseColor.a < data.alphaCutoff)
            {
                ret.alpha = 0; // signal alpha test failed
                return ret;
            }
        }
        ret.alpha = -1; // Negative alpha to signal opaque geometry
    }
    ret.albedo = linearBaseColor.rgb;

    uint metallicRoughnessTex = data.metallicRoughnessTexture & 0xFFFFFF;
    uint metallicRoughnessSampler = data.metallicRoughnessTexture >> 24;
    if (metallicRoughnessTex > 0)
    {
        vec3 mr = texture(
                      sampler2D(
                          materialTextures[metallicRoughnessTex],
                          materialSamplers[metallicRoughnessSampler]),
                      uv)
                      .rgb;
        ret.roughness = mr.g * data.roughnessFactor;
        ret.metallic = mr.b * data.metallicFactor;
    }
    else
    {
        ret.roughness = data.roughnessFactor;
        ret.metallic = data.metallicFactor;
    }

    uint normalTextureTex = data.normalTexture & 0xFFFFFF;
    uint normalTextureSampler = data.normalTexture >> 24;
    if (normalTextureTex > 0)
    {
        vec3 texture_normal = texture(
                                  sampler2D(
                                      materialTextures[normalTextureTex],
                                      materialSamplers[normalTextureSampler]),
                                  uv)
                                  .xyz;
        ret.normal = texture_normal * 2 - 1;
    }
    else
        ret.normal = vec3(-2); // -2 to signal no material normal

    return ret;
}

#endif // MATERIALS_GLSL
