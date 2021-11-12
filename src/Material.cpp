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
    return {_baseColorFactor,
            _metallicFactor,
            _roughnessFactor,
            alphaModeFloat(),
            _alphaCutoff,
            _texCoordSets.baseColor,
            _texCoordSets.metallicRoughness,
            _texCoordSets.normal};
}
