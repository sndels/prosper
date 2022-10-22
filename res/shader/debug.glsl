#ifndef DEBUG_GLSL
#define DEBUG_GLSL

#include "material.glsl"
#include "random.glsl"

const uint CommonDebugDrawType_PrimitiveID = 0;
const uint CommonDebugDrawType_MeshID = 1;
const uint CommonDebugDrawType_MaterialID = 2;
const uint CommonDebugDrawType_Albedo = 3;
const uint CommonDebugDrawType_ShadingNormal = 4;
const uint CommonDebugDrawType_Roughness = 5;
const uint CommonDebugDrawType_Metallic = 6;

struct DebugInputs
{
    uint meshID;
    uint primitiveID;
    uint materialID;
    vec3 normal;
};

vec3 commonDebugDraw(uint drawType, DebugInputs inputs, Material m)
{
    if (drawType == CommonDebugDrawType_PrimitiveID)
        return vec3(uintToColor(inputs.primitiveID));
    else if (drawType == CommonDebugDrawType_MeshID)
        return vec3(uintToColor(inputs.meshID));
    else if (drawType == CommonDebugDrawType_MaterialID)
        return vec3(uintToColor(inputs.materialID));
    else if (drawType == CommonDebugDrawType_Albedo)
        return vec3(m.albedo);
    else if (drawType == CommonDebugDrawType_ShadingNormal)
        return vec3(inputs.normal * 0.5 + 0.5);
    else if (drawType == CommonDebugDrawType_Roughness)
        return vec3(vec3(m.roughness));
    else if (drawType == CommonDebugDrawType_Metallic)
        return vec3(vec3(m.metallic));
    return vec3(1, 0, 1);
}

#endif // DEBUG_GLSL
