#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_CULLER_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_CULLER_H

#include "../fields.h"

struct DrawListCullerPC
{
    STRUCT_FIELD_GLM(uvec2, hizResolution, {});
    STRUCT_FIELD_GLM(vec2, hizUvScale, 1.f);
    // 0 means no hiz bound
    STRUCT_FIELD_GLM(uint, hizMipCount, 0);
    STRUCT_FIELD_GLM(uint, outputSecondPhaseInput, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_DRAW_LIST_CULLER_H
