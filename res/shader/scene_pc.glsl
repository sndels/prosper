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

#endif // SCENE_PC_GLSL
