#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_TONE_MAP_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_TONE_MAP_H

#include "../fields.h"

struct ToneMapPC
{
    STRUCT_FIELD(float, exposure, 1.f);
    STRUCT_FIELD(float, contrast, 1.f);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_TONE_MAP_H
