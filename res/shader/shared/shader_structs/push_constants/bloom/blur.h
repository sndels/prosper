#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_BLUR_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_BLUR_H

#include "../../fields.h"

struct BlurPC
{
    STRUCT_FIELD_GLM(uvec2, resolution, 0);
    STRUCT_FIELD_GLM(vec2, invResolution, 0.f);
    STRUCT_FIELD_GLM(uint, mipLevel, 0);
    STRUCT_FIELD_GLM(uint, transpose, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_BLUR_H
