#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DOF_REDUCE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DOF_REDUCE_H

#include "../../fields.h"

struct ReducePC
{
    STRUCT_FIELD_GLM(ivec2, topMipResolution, {});
    STRUCT_FIELD_GLM(uint, numWorkGroupsPerSlice, 0);
    STRUCT_FIELD_GLM(uint, mips, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DOF_REDUCE_H
