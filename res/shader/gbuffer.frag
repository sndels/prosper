#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "scene/camera.glsl"
#include "scene/instances.glsl"
#include "scene/materials.glsl"

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in float inZCam;
layout(location = 2) in vec2 inTexCoord0;
layout(location = 3) in vec4 inPositionNDC;
layout(location = 4) in vec4 inPrevPositionNDC;
layout(location = 5) in mat3 inTbn;
layout(location = 8) in flat uint inDrawInstanceID;
layout(location = 9) in flat uint inMeshletID;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalMetallic;
layout(location = 2) out vec2 outVelocity;

mat3 generateTBN()
{
    // http://www.thetenthplanet.de/archives/1180
    vec3 dp1 = dFdx(inPositionWorld);
    vec3 dp2 = dFdy(inPositionWorld);
    vec2 duv1 = dFdx(inTexCoord0);
    vec2 duv2 = dFdy(inTexCoord0);

    vec3 N = normalize(inTbn[2]);
    vec3 T = normalize(dp1 * duv2.t - dp2 * duv1.t);
    vec3 B = normalize(cross(N, T));
    return mat3(T, B, N);
}

void main()
{
    DrawInstance instance = drawInstances.instance[inDrawInstanceID];
    Material material = sampleMaterial(instance.materialID, inTexCoord0);

    // Early out if alpha test failed / zero alpha
    if (material.alpha == 0)
        discard;

    vec3 normal;
    if (material.normal.x != -2) // -2 signals no material normal
    {
        mat3 TBN = length(inTbn[0]) > 0 ? inTbn : generateTBN();
        normal = normalize(TBN * material.normal.xyz);
    }
    else
        normal = normalize(inTbn[2]);

    // Store in NDC like in https://alextardif.com/TAA.html
    vec3 posNDC = inPositionNDC.xyz / inPositionNDC.w;
    vec3 prevPosNDC = inPrevPositionNDC.xyz / inPrevPositionNDC.w;
    vec2 velocity = (posNDC.xy - camera.currentJitter) -
                    (prevPosNDC.xy - camera.previousJitter);
    // Let's have positive motion be upward in the image to try and avoid
    // confusion.
    velocity.y = -velocity.y;

    // TODO:
    // Does GLSL support passing uniforms as parameters some way?
    // G-Buffer packing should be a function/macro
    outAlbedoRoughness = vec4(material.albedo, material.roughness);
    outNormalMetallic = vec4(normal, material.metallic);
    outVelocity = clamp(velocity, vec2(-1), vec2(1));
    // No alpha needed as only opaque surfaces are in gbuffer
}
