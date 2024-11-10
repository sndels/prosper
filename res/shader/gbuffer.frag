#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "common/math.glsl"
#include "debug.glsl"
#include "scene/camera.glsl"
#include "scene/instances.glsl"
#include "scene/materials.glsl"
#include "shared/shader_structs/push_constants/gbuffer.h"

layout(push_constant) uniform PushConstants { GBufferPC PC; };

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in float inZCam;
layout(location = 2) in vec2 inTexCoord0;
layout(location = 3) in vec4 inPositionNDC;
layout(location = 4) in vec4 inPrevPositionNDC;
layout(location = 5) in vec3 inNormalWorld;
layout(location = 6) in vec4 inTangentWorldSign;
layout(location = 7) in flat uint inDrawInstanceIndex;
layout(location = 8) in flat uint inMeshletIndex;

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
    DrawInstance instance = drawInstances.instance[inDrawInstanceIndex];
    Material material = sampleMaterial(instance.materialIndex, inTexCoord0);

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

    // Write before debug to not break TAA
    outVelocity = clamp(velocity, vec2(-1), vec2(1));

    if (PC.drawType != DrawType_Default)
    {
        if (PC.drawType == DrawType_MeshletID)
        {
            outAlbedoRoughness = vec4(uintToColor(inMeshletIndex), 1);
            outNormalMetallic = vec4(0);
            return;
        }

        DebugInputs di;
        di.meshIndex = instance.meshIndex;
        di.primitiveID = gl_PrimitiveID;
        di.materialIndex = instance.materialIndex;
        di.shadingNormal = normal;
        di.texCoord0 = inTexCoord0;
        outAlbedoRoughness =
            vec4(commonDebugDraw(PC.drawType, di, material), 1);
        outNormalMetallic = vec4(0);
        return;
    }

    outAlbedoRoughness = vec4(material.albedo, material.roughness);
    outNormalMetallic =
        vec4(encodedNormal.xy, material.metallic, encodedNormal.z);
}
