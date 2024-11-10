#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_RT_TRACE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_RT_TRACE_H

#include "../../fields.h"

struct TracePC
{
    STRUCT_FIELD_GLM(uint, drawType, 0);
    STRUCT_FIELD_GLM(uint, frameIndex, 0);
    STRUCT_FIELD_GLM(uint, flags, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_RT_TRACE_H
