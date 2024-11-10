#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_HIZ_DOWNSAMPLER_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_HIZ_DOWNSAMPLER_H

#include "../fields.h"

struct HizDownsamplerPC
{
    STRUCT_FIELD_GLM(ivec2, inputResolution, {});
    STRUCT_FIELD_GLM(ivec2, topMipResolution, {});
    STRUCT_FIELD_GLM(uint, numWorkGroupsPerSlice, 0);
    STRUCT_FIELD_GLM(uint, mips, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_HIZ_DOWNSAMPLER_H
