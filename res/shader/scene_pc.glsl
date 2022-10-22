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

#define CommonDebugDrawTypeDefault 0
#define CommonDebugDrawTypeOffset 1

#endif // SCENE_PC_GLSL
