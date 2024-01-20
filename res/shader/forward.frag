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
#include "scene/light_clusters.glsl"
#include "scene/lighting.glsl"
#include "scene/lights.glsl"
#include "scene/materials.glsl"
#include "scene/skybox.glsl"

layout(location = 0) in InVertex
{
    vec3 positionWorld;
    float zCam;
    vec2 texCoord0;
    vec4 positionNDC;
    vec4 prevPositionNDC;
    mat3 tbn;
}
inVertex;

layout(location = 9) in InPrimitive { flat uint meshletID; }
inPrimitive;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;

mat3 generateTBN()
{
    // http://www.thetenthplanet.de/archives/1180
    vec3 dp1 = dFdx(inVertex.positionWorld);
    vec3 dp2 = dFdy(inVertex.positionWorld);
    vec2 duv1 = dFdx(inVertex.texCoord0);
    vec2 duv2 = dFdy(inVertex.texCoord0);

    vec3 N = normalize(inVertex.tbn[2]);
    vec3 T = normalize(dp1 * duv2.t - dp2 * duv1.t);
    vec3 B = normalize(cross(N, T));
    return mat3(T, B, N);
}

void main()
{
    VisibleSurface surface;
    surface.positionWS = inVertex.positionWorld;
    surface.invViewRayWS = normalize(camera.eye.xyz - inVertex.positionWorld);
    surface.uv = inVertex.texCoord0;
    surface.material = sampleMaterial(PC.MaterialID, inVertex.texCoord0);

    // Early out if alpha test failed / zero alpha
    if (surface.material.alpha == 0)
        discard;

    if (surface.material.normal.x != -2) // -2 signals no material normal
    {
        mat3 TBN = length(inVertex.tbn[0]) > 0 ? inVertex.tbn : generateTBN();
        surface.normalWS = normalize(TBN * surface.material.normal.xyz);
    }
    else
        surface.normalWS = normalize(inVertex.tbn[2]);

    surface.NoV = saturate(dot(surface.normalWS, surface.invViewRayWS));

    vec3 color = vec3(0);

    color += evalDirectionalLight(surface);

    LightClusterInfo lightInfo =
        unpackClusterPointer(uvec2(gl_FragCoord.xy), inVertex.zCam);

    color += evalPointLights(surface, lightInfo);

    color += evalSpotLights(surface, lightInfo);

    if (PC.ibl == 1)
        color += evalIBL(surface);

    float alpha = surface.material.alpha > 0 ? surface.material.alpha : 1.0;

    // Store in NDC like in https://alextardif.com/TAA.html
    vec3 posNDC = inVertex.positionNDC.xyz / inVertex.positionNDC.w;
    vec3 prevPosNDC = inVertex.prevPositionNDC.xyz / inVertex.prevPositionNDC.w;
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
            outColor = vec4(uintToColor(inPrimitive.meshletID), 1);
            return;
        }

        DebugInputs di;
        di.meshID = PC.MeshID;
        di.primitiveID = gl_PrimitiveID;
        di.materialID = PC.MaterialID;
        di.position = surface.positionWS;
        di.shadingNormal = surface.normalWS;
        di.texCoord0 = inVertex.texCoord0;
        outColor = vec4(commonDebugDraw(PC.DrawType, di, surface.material), 1);
        return;
    }

    outColor = vec4(color, alpha);
}
