#ifndef RANDOM_GLSL
#define RANDOM_GLSL

// From Supplement to Hash Functions for GPU Rendering
// By Jarzynski & Olano
// https://jcgt.org/published/0009/03/02/supplementary.pdf
uint pcg(uint v)
{
    uint state = v * 747796405 + 2891336453;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

vec3 uintToColor(uint x)
{
    // Hashing should, on average, favor pastellish colors that are easy on the
    // eyes. Numerically and spatially adjacent indices should also have a
    // decent probability of getting distinct colors.
    uint xr = pcg(x);
    uint r = (xr >> 20) & 0x3FF;
    uint g = (xr >> 10) & 0x3FF;
    uint b = xr & 0x3FF;
    return vec3(r, g, b) / 0x3FF;
}

#endif // RANDOM_GLSL
