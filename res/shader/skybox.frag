#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable

#include "scene/skybox.glsl"

layout(location = 0) in vec3 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 color = textureLod(skybox, fragTexCoord, 0).rgb;
    outColor = vec4(color, 1);
}
