#ifndef SCENE_INTSTANCES_GLSL
#define SCENE_INTSTANCES_GLSL

#include "../shared/shader_structs/scene/draw_instance.h"
#include "../shared/shader_structs/scene/model_instance_transforms.h"
#include "vertex.glsl"

layout(std430, set = SCENE_INSTANCES_SET, binding = 0) readonly buffer
    ModelInstanceTransformsDSB
{
    ModelInstanceTransforms instance[];
}
modelInstanceTransforms;

layout(std430, set = SCENE_INSTANCES_SET, binding = 1) readonly buffer
    PreviousModelInstanceTransformsDSB
{
    ModelInstanceTransforms instance[];
}
previousModelInstanceTransforms;

layout(std430, set = SCENE_INSTANCES_SET, binding = 2) readonly buffer
    ModelInstanceScalesDSB
{
    float instance[];
}
modelInstanceScales;

layout(std430, set = SCENE_INSTANCES_SET, binding = 3) readonly buffer
    DrawInstances
{
    DrawInstance instance[];
}
drawInstances;

Vertex transform(Vertex v, ModelInstanceTransforms t)
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

vec3 worldPosition(Vertex v, ModelInstanceTransforms t)
{
    // 3x4 SRT multiplies from the right
    return (vec4(v.Position, 1.0) * t.modelToWorld).xyz;
}

#endif // SCENE_INTSTANCES_GLSL
