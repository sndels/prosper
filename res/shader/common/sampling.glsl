#ifndef COMMON_SAMPLING_GLSL
#define COMMON_SAMPLING_GLSL

#include "math.glsl"

// From 'Sampling Transformations Zoo'
// by Peter Shirley et al.
// Published in Ray Tracing Gems
vec3 cosineSampleHemisphere(vec3 n, vec2 u)
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

    return normalize(n + vec3(x, y, z));
}

float cosineHemispherePdf(float NoL) { return NoL / PI; }

mat3 orthonormalBasis(vec3 n)
{
    // From Building an Orthonormal Basis, Revisited
    // By Duff et al.
    float s = sign(n.z);
    float a = -1. / (s + n.z);
    float b = n.x * n.y * a;
    vec3 b1 = vec3(1. + s * n.x * n.x * a, s * b, -s * n.x);
    vec3 b2 = vec3(b, s + n.y * n.y * a, -n.y);
    return transpose(mat3(b1, b2, n));
}

#endif // COMMON_SAMPLING_GLSL
