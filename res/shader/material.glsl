#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct Material
{
    // albedo.r < 0 will signal that alpha was under cutoff
    vec3 albedo;
    // normal.x == -2 will signal that material doesn't include a surface normal
    vec3 normal;
    float roughness;
    float metallic;
    // alpha < 0 will signal opaque
    // alpha == 0 will signal alpha was under cutoff (or blend value was 0)
    // alpha > 0 will signal alpha testing should be used
    float alpha;
};

#endif // MATERIAL_GLSL