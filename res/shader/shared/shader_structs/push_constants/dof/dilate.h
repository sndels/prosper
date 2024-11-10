#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DOF_DILATE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DOF_DILATE_H

#include "../../fields.h"

struct DilatePC
{
    STRUCT_FIELD_GLM(ivec2, res, {});
    STRUCT_FIELD_GLM(vec2, invRes, {});
    STRUCT_FIELD(int, gatherRadius, 1);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DOF_DILATE_H
