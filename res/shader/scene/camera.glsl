#ifndef SCENE_CAMERA_GLSL
#define SCENE_CAMERA_GLSL

#include "../shared/shader_structs/scene/camera.h"

layout(std430, set = CAMERA_SET, binding = 0) readonly buffer CameraDSB
{
    CameraUniforms camera;
};

float linearizeDepth(float nonLinearDepth)
{
    // vec4 tmp = inverse(camera.cameraToClip) * vec4(0, 0, depth, 1);
    // return tmp.z / tmp.w;
    // TODO:
    // This is "optimized" version is different for orthographic for e.g. sun
    // shadows. Ortho should be detectable by proj[3][3] or the linearization
    // could be parametrized differently and stored in the same constants for
    // both projection types
    return -camera.cameraToClip[3][2] /
           (nonLinearDepth + camera.cameraToClip[2][2]);
}

vec3 worldPos(vec2 uv, float nonLinearDepth)
{
    // TODO: This is broken
    // TODO: Cheaper view ray
    vec4 surfaceVS = camera.clipToWorld * vec4(uv * 2 - 1, nonLinearDepth, 1);
    return surfaceVS.xyz / surfaceVS.w;
}

vec3 cameraWorldFwd()
{
    return -vec3(
        camera.worldToCamera[0].z, camera.worldToCamera[1].z,
        camera.worldToCamera[2].z);
}

#endif // SCENE_CAMERA_GLSL
