#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_COMPOSE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_COMPOSE_H

#include "../../fields.h"

struct ComposePC
{
    STRUCT_FIELD_GLM(vec2, illuminationResolution, 0.f);
    STRUCT_FIELD_GLM(vec2, invIlluminationResolution, 0.f);
    STRUCT_FIELD_GLM(vec3, blendFactors, 0.f);
    STRUCT_FIELD(float, invBloomDimSquared, 0.f);
    STRUCT_FIELD_GLM(uint, resolutionScale, 2);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_COMPOSE_H
