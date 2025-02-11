#ifndef SCENE_MATERIALS_GLSL
#define SCENE_MATERIALS_GLSL

#include "../shared/shader_structs/scene/material_data.h"
#include "material.glsl"

layout(std430, set = MATERIAL_DATAS_SET, binding = 0) readonly buffer
    MaterialDatas
{
    MaterialData materials[];
}
materialDatas;

layout(std430, set = MATERIAL_DATAS_SET, binding = 1) readonly buffer
    GlobalMaterialConstantsDSB
{
    float lodBias;
}
globalMaterialConstants;

layout(set = MATERIAL_TEXTURES_SET, binding = 0) uniform sampler
    materialSamplers[NUM_MATERIAL_SAMPLERS];
layout(set = MATERIAL_TEXTURES_SET, binding = NUM_MATERIAL_SAMPLERS)
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

#define GET_MATERIAL_TEXTURE(index) materialTextures[nonuniformEXT(index)]
#define GET_MATERIAL_SAMPLER(index) materialSamplers[nonuniformEXT(index)]

#ifdef USE_MATERIAL_LOD_BIAS
#define sampleMaterialTexture(smplr, uv)                                       \
    texture(smplr, uv, globalMaterialConstants.lodBias)
#else // !USE_MATERIAL_LOD_BIAS
#define sampleMaterialTexture(smplr, uv) texture(smplr, uv)
#endif // USE_MATERIAL_LOD_BIAS

Material sampleMaterial(uint index, vec2 uv)
{
    MaterialData data = materialDatas.materials[index];
    Material ret;

    vec4 linearBaseColor;
    uint baseColorTex = data.baseColorTextureSampler & 0xFFFFFF;
    uint baseColorSampler = data.baseColorTextureSampler >> 24;
    if (baseColorTex > 0)
        linearBaseColor = sRGBtoLinear(sampleMaterialTexture(
            sampler2D(
                GET_MATERIAL_TEXTURE(baseColorTex),
                GET_MATERIAL_SAMPLER(baseColorSampler)),
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

    uint metallicRoughnessTex = data.metallicRoughnessTextureSampler & 0xFFFFFF;
    uint metallicRoughnessSampler = data.metallicRoughnessTextureSampler >> 24;
    if (metallicRoughnessTex > 0)
    {
        vec3 mr = sampleMaterialTexture(
                      sampler2D(
                          GET_MATERIAL_TEXTURE(metallicRoughnessTex),
                          GET_MATERIAL_SAMPLER(metallicRoughnessSampler)),
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
    // Avoid losing specular at zero roughness
    ret.roughness = max(ret.roughness, 0.05);

    uint normalTextureTex = data.normalTextureSampler & 0xFFFFFF;
    uint normalTextureSampler = data.normalTextureSampler >> 24;
    if (normalTextureTex > 0)
    {
        vec3 texture_normal =
            sampleMaterialTexture(
                sampler2D(
                    GET_MATERIAL_TEXTURE(normalTextureTex),
                    GET_MATERIAL_SAMPLER(normalTextureSampler)),
                uv)
                .xyz;
        ret.normal = texture_normal * 2 - 1;
    }
    else
        ret.normal = vec3(-2); // -2 to signal no material normal

    return ret;
}

float sampleAlpha(uint index, vec2 uv)
{
    MaterialData data = materialDatas.materials[index];

    float linearAlpha = 1;
    uint baseColorTex = data.baseColorTextureSampler & 0xFFFFFF;
    uint baseColorSampler = data.baseColorTextureSampler >> 24;
    if (baseColorTex > 0)
        linearAlpha =
            sRGBtoLinear(sampleMaterialTexture(
                             sampler2D(
                                 GET_MATERIAL_TEXTURE(baseColorTex),
                                 GET_MATERIAL_SAMPLER(baseColorSampler)),
                             uv)
                             .a);
    linearAlpha *= data.baseColorFactor.a;

    if (data.alphaMode == AlphaModeBlend)
        return linearAlpha;

    if (data.alphaMode == AlphaModeMask)
    {
        if (linearAlpha < data.alphaCutoff)
            return 0; // signal alpha test failed
    }
    return -1; // Negative alpha to signal opaque geometry
}

#endif // SCENE_MATERIALS_GLSL
