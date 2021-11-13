#include "Material.hpp"

float Material::alphaModeFloat() const {
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

Material::PCBlock Material::pcBlock() const {
    return PCBlock{.baseColorFactor = _baseColorFactor,
                   .metallicFactor = _metallicFactor,
                   .roughnessFactor = _roughnessFactor,
                   .alphaMode = alphaModeFloat(),
                   .alphaCutoff = _alphaCutoff,
                   .baseColorTextureSet = _texCoordSets.baseColor,
                   .metallicRoughnessTextureSet =
                       _texCoordSets.metallicRoughness,
                   .normalTextureSet = _texCoordSets.normal};
}
