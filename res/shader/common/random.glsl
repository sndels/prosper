#ifndef COMMON_RANDOM_GLSL
#define COMMON_RANDOM_GLSL

// From Supplement to Hash Functions for GPU Rendering
// By Jarzynski & Olano
// https://jcgt.org/published/0009/03/02/supplementary.pdf
uint pcg(uint v)
{
    uint state = v * 747796405 + 2891336453;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

// From Hash Functions for GPU Rendering
// By Jarzynski & Olano
// https://jcgt.org/published/0009/03/02/supplementary.pdf
uvec3 pcg3d(uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
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

float rngTo01(uint u) { return u / float(0xFFFFFFFFu); }

vec2 rngTo01(vec2 u) { return u / float(0xFFFFFFFFu); }

vec3 rngTo01(vec3 u) { return u / float(0xFFFFFFFFu); }

// Should be initialized at the shader entrypoint e.g. as uvec3(px, frameIndex)
uvec3 pcg_state;
float rnd01()
{
    // TODO: Verify this doesn't break subsequent pcg3d samples
    pcg_state.x = pcg(pcg_state.x);
    return rngTo01(pcg_state.x);
}
vec2 rnd2d01()
{
    pcg_state = pcg3d(pcg_state);
    return rngTo01(pcg_state.xy);
}
vec3 rnd3d01()
{
    pcg_state = pcg3d(pcg_state);
    return rngTo01(pcg_state.xyz);
}

#endif // COMMON_RANDOM_GLSL
