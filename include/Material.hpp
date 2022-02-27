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
    uint32_t baseColor{0};
    uint32_t metallicRoughness{0};
    uint32_t normal{0};
    uint32_t pad{0};
};

#endif // PROSPER_MATERIAL_HPP
