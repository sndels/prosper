#ifndef DEBUG_GLSL
#define DEBUG_GLSL

#include "common/random.glsl"
#include "scene/material.glsl"

struct DebugInputs
{
    uint meshIndex;
    uint primitiveID;
    uint materialIndex;
    vec3 position;
    vec3 shadingNormal;
    vec2 texCoord0;
};

vec3 commonDebugDraw(uint drawType, DebugInputs inputs, Material m)
{
    if (drawType == DrawType_PrimitiveID)
        return vec3(uintToColor(inputs.primitiveID));
    else if (drawType == DrawType_MeshID)
        return vec3(uintToColor(inputs.meshIndex));
    else if (drawType == DrawType_MaterialID)
        return vec3(uintToColor(inputs.materialIndex));
    else if (drawType == DrawType_Position)
        return vec3(inputs.position);
    else if (drawType == DrawType_TexCoord0)
        return vec3(inputs.texCoord0, 0);
    else if (drawType == DrawType_Albedo)
        return vec3(m.albedo);
    else if (drawType == DrawType_ShadingNormal)
        return vec3(inputs.shadingNormal * 0.5 + 0.5);
    else if (drawType == DrawType_Roughness)
        return vec3(vec3(m.roughness));
    else if (drawType == DrawType_Metallic)
        return vec3(vec3(m.metallic));
    return vec3(1, 0, 1);
}

#endif // DEBUG_GLSL
