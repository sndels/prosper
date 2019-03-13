#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform Transforms {
    mat4 modelToClip;
} transforms;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = transforms.modelToClip * vec4(pos, 1.0);
    fragColor = color;
}
