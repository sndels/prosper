#ifndef SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_FFT_H
#define SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_FFT_H

#include "../../fields.h"

struct FftPC
{
    STRUCT_FIELD_GLM(uvec2, inputResolution, {});
    STRUCT_FIELD_GLM(uvec2, outputResolution, {});
    STRUCT_FIELD_GLM(uint, ns, 0);
    STRUCT_FIELD_GLM(uint, r, 0);
    STRUCT_FIELD_GLM(uint, flags, 0);
};

#endif // SHADER_STRUCTS_PUSH_CONSTANTS_BLOOM_FFT_H
