#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_GENERATOR_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_GENERATOR_H

#include "../fields.h"

struct DrawListGeneratorPC
{
    STRUCT_FIELD_GLM(uint, matchTransparents, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_GENERATOR_H