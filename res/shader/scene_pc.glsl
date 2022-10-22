#ifndef SCENE_PC_GLSL
#define SCENE_PC_GLSL

layout(push_constant) uniform ScenePC
{
    uint ModelInstanceID;
    uint MeshID;
    uint MaterialID;
    uint DrawType;
}
scenePC;

const uint DrawType_Default = 0;
const uint DrawType_PrimitiveID = 1;
const uint DrawType_MeshID = 2;
const uint DrawType_MaterialID = 3;
const uint DrawType_Albedo = 4;
const uint DrawType_ShadingNormal = 5;
const uint DrawType_Roughness = 6;
const uint DrawType_Metallic = 7;

#endif // SCENE_PC_GLSL
