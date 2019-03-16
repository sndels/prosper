#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform Camera {
    mat4 worldToCamera;
    mat4 cameraToClip;
} camera;

/*
layout(set = 1, binding = 0) uniform Object {
    mat4 modelToWorld;
} object;
*/

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = camera.cameraToClip *
                  camera.worldToCamera *
                  //object.modelToWorld *
                  vec4(pos, 1.0);
    fragColor = color;
}
