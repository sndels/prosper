#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require

#include "brdf.glsl"
#include "common/random.glsl"
#include "debug.glsl"
#include "forward_pc.glsl"
#include "scene/camera.glsl"
#include "scene/instances.glsl"
#include "scene/light_clusters.glsl"
#include "scene/lighting.glsl"
#include "scene/lights.glsl"
#include "scene/materials.glsl"
#include "scene/skybox.glsl"

layout(location = 0) in vec3 inPositionWorld;
layout(location = 1) in float inZCam;
layout(location = 2) in vec2 inTexCoord0;
layout(location = 3) in vec4 inPositionNDC;
layout(location = 4) in vec4 inPrevPositionNDC;
layout(location = 5) in vec3 inNormalWorld;
layout(location = 6) in vec4 inTangentWorldSign;
layout(location = 7) in flat uint inDrawInstanceID;
layout(location = 8) in flat uint inMeshletID;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;

vec3 mappedNormal(vec3 tangentSpaceNormal, vec3 normal, vec3 tangent, float sgn)
{
    vec3 vNt = tangentSpaceNormal;
    vec3 vN = normal;
    vec3 vT = tangent;
    // From mikktspace.com
    vec3 vB = sgn * cross(vN, vT);
    return normalize(vNt.x * vT + vNt.y * vB + vNt.z * vN);
}

void main()
{
    DrawInstance instance = drawInstances.instance[inDrawInstanceID];

    VisibleSurface surface;
    surface.positionWS = inPositionWorld;
    surface.invViewRayWS = normalize(camera.eye.xyz - inPositionWorld);
    surface.uv = inTexCoord0;
    surface.material = sampleMaterial(instance.materialID, inTexCoord0);

    // Early out if alpha test failed / zero alpha
    if (surface.material.alpha == 0)
        discard;

    if (surface.material.normal.x != -2) // -2 signals no material normal
        surface.normalWS = mappedNormal(
            surface.material.normal, inNormalWorld, inTangentWorldSign.xyz,
            inTangentWorldSign.w);
    else
        surface.normalWS = normalize(inNormalWorld);

    surface.NoV = saturate(dot(surface.normalWS, surface.invViewRayWS));

    vec3 color = vec3(0);

    color += evalDirectionalLight(surface);

    LightClusterInfo lightInfo =
        unpackClusterPointer(uvec2(gl_FragCoord.xy), inZCam);

    color += evalPointLights(surface, lightInfo);

    color += evalSpotLights(surface, lightInfo);

    if (PC.ibl == 1)
        color += evalIBL(surface);

    float alpha = surface.material.alpha > 0 ? surface.material.alpha : 1.0;

    // Store in NDC like in https://alextardif.com/TAA.html
    vec3 posNDC = inPositionNDC.xyz / inPositionNDC.w;
    vec3 prevPosNDC = inPrevPositionNDC.xyz / inPrevPositionNDC.w;
    vec2 velocity = (posNDC.xy - camera.currentJitter) -
                    (prevPosNDC.xy - camera.previousJitter);
    // Let's have positive motion be upward in the image to try and avoid
    // confusion.
    velocity.y = -velocity.y;

    outVelocity = clamp(velocity, vec2(-1), vec2(1));

    if (PC.DrawType >= DrawType_PrimitiveID)
    {
        if (PC.DrawType == DrawType_MeshletID)
        {
            outColor = vec4(uintToColor(inMeshletID), 1);
            return;
        }

        DebugInputs di;
        di.meshID = instance.meshID;
        di.primitiveID = gl_PrimitiveID;
        di.materialID = instance.materialID;
        di.position = surface.positionWS;
        di.shadingNormal = surface.normalWS;
        di.texCoord0 = inTexCoord0;
        outColor = vec4(commonDebugDraw(PC.DrawType, di, surface.material), 1);
        return;
    }

    outColor = vec4(color, alpha);
}
