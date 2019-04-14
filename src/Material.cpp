#include "Material.hpp"

Material::Material(Material&& other) :
    _baseColor(other._baseColor),
    _metallicRoughness(other._metallicRoughness),
    _normal(other._normal),
    _texCoordSets(other._texCoordSets),
    _baseColorFactor(other._baseColorFactor),
    _metallicFactor(other._metallicFactor),
    _roughnessFactor(other._roughnessFactor),
    _alphaMode(other._alphaMode),
    _alphaCutoff(other._alphaCutoff),
    _descriptorSet(other._descriptorSet)
{ }

float Material::alphaModeFloat() const
{
    switch (_alphaMode) {
    case AlphaMode::Opaque:
        return 0.f;
    case AlphaMode::Mask:
        return 1.f;
    case AlphaMode::Blend:
        return 2.f;
    default:
        throw std::runtime_error("Unimplemented AlphaMode->float -conversion");
    }
    return -1;
}
