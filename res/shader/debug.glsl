#ifndef DEBUG_GLSL
#define DEBUG_GLSL

#include "material.glsl"
#include "random.glsl"

struct DebugInputs
{
    uint meshID;
    uint primitiveID;
    uint materialID;
    vec3 normal;
};

vec3 commonDebugDraw(uint drawType, DebugInputs inputs, Material m)
{
    if (drawType == DrawType_PrimitiveID)
        return vec3(uintToColor(inputs.primitiveID));
    else if (drawType == DrawType_MeshID)
        return vec3(uintToColor(inputs.meshID));
    else if (drawType == DrawType_MaterialID)
        return vec3(uintToColor(inputs.materialID));
    else if (drawType == DrawType_Albedo)
        return vec3(m.albedo);
    else if (drawType == DrawType_ShadingNormal)
        return vec3(inputs.normal * 0.5 + 0.5);
    else if (drawType == DrawType_Roughness)
        return vec3(vec3(m.roughness));
    else if (drawType == DrawType_Metallic)
        return vec3(vec3(m.metallic));
    return vec3(1, 0, 1);
}

#endif // DEBUG_GLSL
