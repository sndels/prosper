#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "common/math.glsl"
#include "scene/camera.glsl"
#include "scene/instances.glsl"
#include "scene/materials.glsl"

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in float inZCam;
layout(location = 2) in vec2 inTexCoord0;
layout(location = 3) in vec4 inPositionNDC;
layout(location = 4) in vec4 inPrevPositionNDC;
layout(location = 5) in vec3 inNormalWorld;
layout(location = 6) in vec4 inTangentWorldSign;
layout(location = 7) in flat uint inDrawInstanceID;
layout(location = 8) in flat uint inMeshletID;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalMetallic;
layout(location = 2) out vec2 outVelocity;

vec3 mappedNormal(vec3 tangentSpaceNormal, vec3 normal, vec3 tangent, float sgn)
{
    vec3 vNt = tangentSpaceNormal;
    vec3 vN = normal;
    vec3 vT = tangent;
    // From mikktspace.com
    vec3 vB = sgn * cross(vN, vT);
    return normalize(vNt.x * vT + vNt.y * vB + vNt.z * vN);
}

// Adapted from
// https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
vec3 signedOctEncode(vec3 n)
{
    vec3 OutN;

    n /= (abs(n.x) + abs(n.y) + abs(n.z));

    OutN.y = n.y * 0.5 + 0.5;
    OutN.x = n.x * 0.5 + OutN.y;
    OutN.y = n.x * -0.5 + OutN.y;

    const float fltMax = 3.40282e+38;
    OutN.z = saturate(n.z * fltMax);
    return OutN;
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
        normal = mappedNormal(
            material.normal, inNormalWorld, inTangentWorldSign.xyz,
            inTangentWorldSign.w);
    else
        normal = normalize(inNormalWorld);
    vec3 encodedNormal = signedOctEncode(normal);

    // Store in NDC like in https://alextardif.com/TAA.html
    vec3 posNDC = inPositionNDC.xyz / inPositionNDC.w;
    vec3 prevPosNDC = inPrevPositionNDC.xyz / inPrevPositionNDC.w;
    vec2 velocity = (posNDC.xy - camera.currentJitter) -
                    (prevPosNDC.xy - camera.previousJitter);
    // Let's have positive motion be upward in the image to try and avoid
    // confusion.
    velocity.y = -velocity.y;

    outAlbedoRoughness = vec4(material.albedo, material.roughness);
    outNormalMetallic =
        vec4(encodedNormal.xy, material.metallic, encodedNormal.z);
    outVelocity = clamp(velocity, vec2(-1), vec2(1));
    // No alpha needed as only opaque surfaces are in gbuffer
}
