#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DOF_GATHER_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DOF_GATHER_H

#include "../../fields.h"

struct GatherPC
{
    STRUCT_FIELD_GLM(ivec2, halfResolution, {});
    STRUCT_FIELD_GLM(vec2, invHalfResolution, {});
    STRUCT_FIELD_GLM(uint, frameIndex, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DOF_GATHER_H
