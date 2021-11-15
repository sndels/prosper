#ifndef PROSPER_MATERIAL_HPP
#define PROSPER_MATERIAL_HPP

#include "vulkan.hpp"
#include <glm/glm.hpp>


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

    float alphaModeFloat() const;
    PCBlock pcBlock() const;
};

#endif // PROSPER_MATERIAL_HPP
