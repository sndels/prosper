#ifndef SHARED_SHADER_STRUCTS_SCENE_CAMERA_H
#define SHARED_SHADER_STRUCTS_SCENE_CAMERA_H

#include "../fields.h"

#ifdef __cplusplus
namespace scene::shader_structs
{
#endif // __cplusplus

struct CameraUniforms
{
    STRUCT_FIELD_GLM(mat4, worldToCamera, {});
    STRUCT_FIELD_GLM(mat4, cameraToWorld, {});
    STRUCT_FIELD_GLM(mat4, cameraToClip, {});
    STRUCT_FIELD_GLM(mat4, clipToWorld, {});
    STRUCT_FIELD_GLM(mat4, previousWorldToCamera, {});
    STRUCT_FIELD_GLM(mat4, previousCameraToClip, {});
    STRUCT_FIELD_GLM(mat4, previousClipToWorld, {});
    STRUCT_FIELD_GLM(vec4, eye, {});
    // These are world space plane normal,distance and normals point into the
    // frustum
    STRUCT_FIELD_GLM(vec4, nearPlane, {});
    STRUCT_FIELD_GLM(vec4, farPlane, {});
    STRUCT_FIELD_GLM(vec4, leftPlane, {});
    STRUCT_FIELD_GLM(vec4, rightPlane, {});
    STRUCT_FIELD_GLM(vec4, topPlane, {});
    STRUCT_FIELD_GLM(vec4, bottomPlane, {});
    STRUCT_FIELD_GLM(uvec2, resolution, {});
    STRUCT_FIELD_GLM(vec2, currentJitter, {});
    STRUCT_FIELD_GLM(vec2, previousJitter, {});
    STRUCT_FIELD(float, near, 0.f);
    STRUCT_FIELD(float, far, 0.f);
    STRUCT_FIELD(float, maxViewScale, 0.f);
};

#ifdef __cplusplus
} //  namespace scene::shader_structs
#endif // __cplusplus

#endif // SHARED_SHADER_STRUCTS_SCENE_CAMERA_H
