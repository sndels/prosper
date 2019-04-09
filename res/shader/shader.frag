#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 2, binding = 0) uniform sampler2D baseColor;
layout(set = 2, binding = 1) uniform sampler2D metallicRoughness;
layout(set = 2, binding = 2) uniform sampler2D normal;

// Needs to match Material::PCBlock
layout(push_constant) uniform Material {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureSet;
    int metallicRoughnessTextureSet;
    int normalTextureSet;
} material;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(baseColor, fragTexCoord) * material.baseColorFactor;
}
