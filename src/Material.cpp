#include "Material.hpp"

Material::Material(Material&& other) :
    _baseColor(other._baseColor),
    _metallicRoughness(other._metallicRoughness),
    _normal(other._normal),
    _texCoordSets(other._texCoordSets),
    _baseColorFactor(other._baseColorFactor),
    _metallicFactor(other._metallicFactor),
    _roughnessFactor(other._roughnessFactor),
    _descriptorSet(other._descriptorSet)
{ }
