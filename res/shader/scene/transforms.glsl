#ifndef SCENE_TRANSFORMS_GLSL
#define SCENE_TRANSFORMS_GLSL

#include "vertex.glsl"

struct Transforms
{
    mat3x4 modelToWorld;
    mat3x4 normalToWorld;
};
layout(std430, set = MODEL_INSTANCE_TRFNS_SET, binding = 0) readonly buffer
    ModelInstanceTransformsDSB
{
    Transforms instance[];
}
modelInstanceTransforms;

layout(std430, set = MODEL_INSTANCE_TRFNS_SET, binding = 1) readonly buffer
    PreviousModelInstanceTransformsDSB
{
    Transforms instance[];
}
previousModelInstanceTransforms;

Vertex transform(Vertex v, Transforms t)
{
    Vertex ret;
    // 3x4 SRT multiplies from the right
    ret.Position = (vec4(v.Position, 1.0) * t.modelToWorld).xyz;
    ret.Normal = normalize(v.Normal * mat3(t.normalToWorld));

    // No point in generating normal basis here if no tangent is supplied
    if (v.Tangent.w != 0)
        ret.Tangent =
            vec4(normalize(v.Tangent.xyz * mat3(t.modelToWorld)), v.Tangent.w);
    else
        ret.Tangent = v.Tangent;

    ret.TexCoord0 = v.TexCoord0;

    return ret;
}

vec3 worldPosition(Vertex v, Transforms t)
{
    // 3x4 SRT multiplies from the right
    return (vec4(v.Position, 1.0) * t.modelToWorld).xyz;
}

mat3 generateTBN(vec3 normal, vec4 tangent)
{
    vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;
    return mat3(tangent.xyz, bitangent, normal);
}

#endif // SCENE_TRANSFORMS_GLSL
