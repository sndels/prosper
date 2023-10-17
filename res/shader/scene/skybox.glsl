#ifndef SCENE_SKYBOX_GLSL
#define SCENE_SKYBOX_GLSL

layout(set = SKYBOX_SET, binding = 0) uniform samplerCube skybox;
layout(set = SKYBOX_SET, binding = 1) uniform samplerCube skyboxIrradiance;

#endif // SCENE_SKYBO_GLSL
