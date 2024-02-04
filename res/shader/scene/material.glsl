#ifndef SCENE_MATERIAL_GLSL
#define SCENE_MATERIAL_GLSL

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

Material loadFromGbuffer(
    ivec2 coord, readonly image2D albedoRoughnessImage,
    readonly image2D normalMetallicImage)
{
    vec4 albedoRoughness = imageLoad(albedoRoughnessImage, coord);
    vec4 normalMetallic = imageLoad(normalMetallicImage, coord);

    Material m;
    m.albedo = albedoRoughness.xyz;
    m.roughness = albedoRoughness.w;
    m.normal = normalize(normalMetallic.xyz);
    m.metallic = normalMetallic.w;
    m.alpha = -1.0;

    return m;
}

#endif // SCENE_MATERIAL_GLSL
