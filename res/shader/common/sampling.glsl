#ifndef COMMON_SAMPLING_GLSL
#define COMMON_SAMPLING_GLSL

#include "math.glsl"

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaUnitDisk
vec2 uniformSampleDisk(vec2 u)
{
    float r = sqrt(u[0]);
    float theta = 2. * PI * u[1];
    return vec2(r * cos(theta), r * sin(theta));
}

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

// Adapted from Sampling the GGX Distribution of Visible Normals
// By Eric Heitz
// Removed anisotropy (single channel alpha, isotropic smith geometry term)
// Calculate light direction and its pdf
vec3 sampleVisibleTrowbridgeReitz(vec3 Ve, float alpha, vec2 Us)
{
    // Section 3.2: transforming the view direction to the hemisphere
    // configuration
    float alphaVx = alpha * Ve.x;
    float alphaVy = alpha * Ve.y;
    vec3 Vh = normalize(vec3(alphaVx, alphaVy, Ve.z));
    // Section 4.1: orthonormal basis (with special case if cross product is
    // zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 =
        lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1, 0, 0);
    vec3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    float r = sqrt(Us[0]);
    float phi = 2.0 * PI * Us[1];
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    vec3 Ne = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));

    return reflect(-Ve, Ne);
}

float visibleTrowbridgeReitzPdf(vec3 Ve, vec3 Le, float alpha)
{
    vec3 N = vec3(0, 0, 1);
    vec3 Ne = normalize(Ve + Le);
    float NoV = saturate(dot(N, Ve));
    float NoL = saturate(dot(N, Le));
    float NoH = saturate(dot(N, Ne));

    float VNDF = schlickTrowbridgeReitz(NoL, NoV, alpha) * NoV *
                 trowbridgeReitz(NoH, alpha) / Ve.z;

    return VNDF / (4 * NoV);
}

// From Real Shading in Unreal Engine 4
// by Brian Karis
vec3 importanceSampleIBLTrowbridgeReitz(vec2 Xi, float alpha, vec3 N)
{
    float Phi = 2 * PI * Xi.x;
    float CosTheta = sqrt((1 - Xi.y) / (1 + (alpha * alpha - 1) * Xi.y));
    float SinTheta = sqrt(1 - CosTheta * CosTheta);

    vec3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;

    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 TangentX = normalize(cross(UpVector, N));
    vec3 TangentY = normalize(cross(N, TangentX));
    // Tangent to world space
    return normalize(TangentX * H.x + TangentY * H.y + N * H.z);
}

#endif // COMMON_SAMPLING_GLSL
