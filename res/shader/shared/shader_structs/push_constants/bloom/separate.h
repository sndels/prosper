#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_SEPARATE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_SEPARATE_H

#include "../../fields.h"

struct SeparatePC
{
    STRUCT_FIELD_GLM(vec2, invInResolution, 0.f);
    STRUCT_FIELD(float, threshold, 0.f);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_SEPARATE_H
