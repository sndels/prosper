#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() { outColor = vec4(fragColor, 1); }
