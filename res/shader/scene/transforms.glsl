#ifndef SCENE_TRANSFORMS_GLSL
#define SCENE_TRANSFORMS_GLSL

struct Transforms
{
    mat4 modelToWorld;
    mat4 normalToWorld;
};
layout(std430, set = MODEL_INSTANCE_TRFNS_SET, binding = 0) readonly buffer
    ModelInstanceTransformsDSB
{
    Transforms instance[];
}
modelInstanceTransforms;

Vertex transform(Vertex v, Transforms t)
{
    Vertex ret;
    ret.Position = (t.modelToWorld * vec4(v.Position, 1.0)).xyz;
    ret.Normal = normalize(mat3(t.normalToWorld) * v.Normal);

    // No point in generating normal basis here if no tangent is supplied
    if (v.Tangent.w != 0)
        ret.Tangent =
            vec4(normalize(mat3(t.modelToWorld) * v.Tangent.xyz), v.Tangent.w);
    else
        ret.Tangent = v.Tangent;

    ret.TexCoord0 = v.TexCoord0;

    return ret;
}

mat3 generateTBN(vec3 normal, vec4 tangent)
{
    vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;
    return mat3(tangent.xyz, bitangent, normal);
}

#endif // SCENE_TRANSFORMS_GLSL
