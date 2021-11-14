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

// From http://filmicworlds.com/blog/filmic-tonemapping-operators/
// https://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
vec3 Uncharted2Tonemap(vec3 color)
{
    float A = 0.15; // Shoulder strength
    float B = 0.50; // Linear strength
    float C = 0.10; // Linear angle
    float D = 0.20; // Toe strength
    float E = 0.02; // Toe numerator
    float F = 0.30; // Toe denominator
    return ((color * (A * color + C * B) + D * E) /
            (color * (A * color + B) + D * F)) -
           E / F;
}

vec3 tonemap(vec3 color)
{
    float exposure = 1.0;
    float gamma = 2.2;
    float linearWhite = 11.2;
    vec3 outcol = Uncharted2Tonemap(color * exposure);
    outcol /= Uncharted2Tonemap(vec3(linearWhite));
    return pow(outcol, vec3(1 / gamma));
}

void main()
{
    vec3 color = sRGBtoLinear(textureLod(skybox, fragTexCoord, 0).rgb);
    outColor = vec4(tonemap(color), 1);
}
