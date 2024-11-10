#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_FORWARD_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_FORWARD_H

#include "../fields.h"

struct ForwardPC
{
    // Some of these are mirrored between this and GBufferPC
    STRUCT_FIELD_GLM(uint, drawType, 0);
    STRUCT_FIELD_GLM(uint, ibl, 0);
    STRUCT_FIELD_GLM(uint, previousTransformValid, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_FORWARD_H
