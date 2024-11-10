#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_RT_REFERENCE_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_RT_REFERENCE_H

#include "../fields.h"

struct ReferencePC
{
    STRUCT_FIELD_GLM(uint, drawType, 0);
    STRUCT_FIELD_GLM(uint, flags, 0);
    STRUCT_FIELD_GLM(uint, frameIndex, 0);
    STRUCT_FIELD(float, apertureDiameter, 0.0001f);
    STRUCT_FIELD(float, focusDistance, 1.f);
    STRUCT_FIELD(float, focalLength, 0.f);
    STRUCT_FIELD_GLM(uint, rouletteStartBounce, 3);
    STRUCT_FIELD_GLM(uint, maxBounces, 3);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_RT_REFERENCE_H
