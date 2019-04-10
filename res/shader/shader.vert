#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform Camera {
    mat4 worldToCamera;
    mat4 cameraToClip;
} camera;

layout(set = 1, binding = 0) uniform Object {
    mat4 modelToWorld;
} object;

layout(location = 0) in vec3 vertPosition;
layout(location = 1) in vec3 vertNormal;
layout(location = 2) in vec2 vertTexCoord0;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord0;

void main() {
    vec4 pos = object.modelToWorld *
               vec4(vertPosition, 1.0);
    gl_Position = camera.cameraToClip *
                  camera.worldToCamera *
                  pos;
    fragPosition = pos.xyz;
    fragNormal = transpose(inverse(mat3(object.modelToWorld))) * vertNormal;
    fragTexCoord0 = vertTexCoord0;
}
