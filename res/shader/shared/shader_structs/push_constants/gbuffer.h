#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_GBUFFER_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_GBUFFER_H

#include "../fields.h"

struct GBufferPC
{
    // Some of these are mirrored between this and ForwardPC
    STRUCT_FIELD_GLM(uint, previousTransformValid, 0);
    STRUCT_FIELD_GLM(uint, drawType, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_GBUFFER_H
