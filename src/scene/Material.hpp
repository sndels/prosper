#ifndef PROSPER_SCENE_MATERIAL_HPP
#define PROSPER_SCENE_MATERIAL_HPP

#include <glm/detail/type_vec4.hpp>
#include <glm/fwd.hpp>

class Texture;

struct Texture2DSampler
{
    uint32_t packed{0};

    Texture2DSampler() = default;
    Texture2DSampler(uint32_t texture, uint32_t sampler)
    : packed{(sampler << 24) | texture} {};

    [[nodiscard]] uint32_t texture() const { return packed & 0xFFFFFF; }
};
static_assert(sizeof(Texture2DSampler) == sizeof(uint32_t));

struct Material
{
    enum class AlphaMode : uint32_t
    {
        Opaque = 0,
        Mask = 1,
        Blend = 2,
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
