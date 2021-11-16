#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform samplerCube skybox;

float sRGBtoLinear(float x)
{
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}
vec3 sRGBtoLinear(vec3 v)
{
    return vec3(sRGBtoLinear(v.r), sRGBtoLinear(v.g), sRGBtoLinear(v.b));
}
// Alpha shouldn't be converted
vec4 sRGBtoLinear(vec4 v) { return vec4(sRGBtoLinear(v.rgb), v.a); }

void main()
{
    vec3 color = sRGBtoLinear(textureLod(skybox, fragTexCoord, 0).rgb);
    outColor = vec4(color, 1);
}
