#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

#include "common.glsl"

// From 'Sampling Transformations Zoo'
// by Peter Shirley et al.
// Published in Ray Tracing Gems
void cosineSampleHemisphere(vec3 n, vec2 u, out vec3 d, out float pdf)
{
    float a = 1.0 - 2.0 * u[0];
    a *= 0.99999; // Try to fix precision issues at grazing angles

    float b = sqrt(1.0 - a * a);
    b *= 0.99999; // Try to fix precision issues at grazing angles

    // Point on unit sphere centered at the tip of the normal
    float phi = 2.0 * PI * u[1];
    float x = b * cos(phi);
    float y = b * sin(phi);
    float z = a;

    d = normalize(n + vec3(x, y, z));
    // TODO: RTG has this as a / PI but the result looks very wrong
    pdf = dot(d, n) / PI;
}

#endif // SAMPLING_GLSL
