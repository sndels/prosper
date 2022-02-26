#ifndef PROSPER_MATERIAL_HPP
#define PROSPER_MATERIAL_HPP

#include "vulkan.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

class Texture;

struct Material
{
    enum class AlphaMode
    {
        Opaque,
        Mask,
        Blend
    };

    // Needs to match shader
    struct PCBlock
    {
        glm::vec4 baseColorFactor;
        float metallicFactor;
        float roughnessFactor;
        float alphaMode;
        float alphaCutoff;
        int baseColorTextureSet;
        int metallicRoughnessTextureSet;
        int normalTextureSet;
    };

    struct TexCoordSets
    {
        int baseColor = -1;
        int metallicRoughness = -1;
        int normal = -1;
    };

    // TODO: More parameters
    Texture *_baseColor = nullptr;
    Texture *_metallicRoughness = nullptr;
    Texture *_normal = nullptr;
    TexCoordSets _texCoordSets;
    glm::vec4 _baseColorFactor = glm::vec4(1);
    float _metallicFactor = 1.f;
    float _roughnessFactor = 1.f;
    AlphaMode _alphaMode = AlphaMode::Opaque;
    float _alphaCutoff = 0.5f;
    vk::DescriptorSet _descriptorSet;

    [[nodiscard]] float alphaModeFloat() const;
    [[nodiscard]] PCBlock pcBlock() const;
};

#endif // PROSPER_MATERIAL_HPP
