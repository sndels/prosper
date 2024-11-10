#ifndef SHARED_SHADER_STRUCTS_SCENE_MATERIAL_DATA_H
#define SHARED_SHADER_STRUCTS_SCENE_MATERIAL_DATA_H

#include "../fields.h"

#ifdef __cplusplus

#include <cstdint>
#include <wheels/assert.hpp>

class Texture;

struct Texture2DSampler
{
    static constexpr uint32_t sMaxTextureIndex = 0xFF'FFFF;
    static constexpr uint32_t sMaxSamplerIndex = 0xFF;

    uint32_t packed{0};

    Texture2DSampler() noexcept = default;
    Texture2DSampler(uint32_t texture, uint32_t sampler)
    : packed{(sampler << 24) | texture}
    {
        WHEELS_ASSERT(texture < sMaxTextureIndex);
        WHEELS_ASSERT(sampler < sMaxSamplerIndex);
    };

    [[nodiscard]] uint32_t texture() const { return packed & sMaxTextureIndex; }
};
static_assert(sizeof(Texture2DSampler) == sizeof(uint32_t));

enum AlphaMode : uint32_t
{
    AlphaMode_Opaque,
    AlphaMode_Mask,
    AlphaMode_Blend,
};

#else // !__cplusplus

const uint AlphaModeOpaque = 0;
const uint AlphaModeMask = 1;
const uint AlphaModeBlend = 2;

#endif // __cplusplus

struct MaterialData
{
    STRUCT_FIELD_GLM(vec4, baseColorFactor, 1.f);
    STRUCT_FIELD(float, metallicFactor, 1.f);
    STRUCT_FIELD(float, roughnessFactor, 1.f);
    STRUCT_FIELD(float, alphaCutoff, 0.5f);
    STRUCT_FIELD_SEPARATE_TYPES(uint, AlphaMode, alphaMode, AlphaMode_Opaque);
    STRUCT_FIELD_SEPARATE_TYPES(
        uint, Texture2DSampler, baseColorTextureSampler, Texture2DSampler{});
    STRUCT_FIELD_SEPARATE_TYPES(
        uint, Texture2DSampler, metallicRoughnessTextureSampler,
        Texture2DSampler{});
    STRUCT_FIELD_SEPARATE_TYPES(
        uint, Texture2DSampler, normalTextureSampler, Texture2DSampler{});
    STRUCT_FIELD_GLM(uint, pad, 0);
};

#endif // SHARED_SHADER_STRUCTS_SCENE_MATERIAL_DATA_H
