#ifndef PROSPER_SCENE_MATERIAL_HPP
#define PROSPER_SCENE_MATERIAL_HPP

#include <glm/glm.hpp>
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

struct Material
{
    enum class AlphaMode : uint32_t
    {
        Opaque,
        Mask,
        Blend,
    };

    // Needs to match shader
    glm::vec4 baseColorFactor{1.f};
    float metallicFactor{1.f};
    float roughnessFactor{1.f};
    float alphaCutoff{0.5f};
    AlphaMode alphaMode{AlphaMode::Opaque};
    Texture2DSampler baseColor;
    Texture2DSampler metallicRoughness;
    Texture2DSampler normal;
    uint32_t pad{0};
};

#endif // PROSPER_SCENE_MATERIAL_HPP
