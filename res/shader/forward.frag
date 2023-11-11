#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

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

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in float fragZCam;
layout(location = 2) in vec2 fragTexCoord0;
layout(location = 3) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

mat3 generateTBN()
{
    // http://www.thetenthplanet.de/archives/1180
    vec3 dp1 = dFdx(fragPosition);
    vec3 dp2 = dFdy(fragPosition);
    vec2 duv1 = dFdx(fragTexCoord0);
    vec2 duv2 = dFdy(fragTexCoord0);

    vec3 N = normalize(fragTBN[2]);
    vec3 T = normalize(dp1 * duv2.t - dp2 * duv1.t);
    vec3 B = normalize(cross(N, T));
    return mat3(T, B, N);
}

void main()
{
    VisibleSurface surface;
    surface.positionWS = fragPosition;
    surface.invViewRayWS = normalize(camera.eye.xyz - fragPosition);
    surface.uv = fragTexCoord0;
    surface.material = sampleMaterial(PC.MaterialID, fragTexCoord0);

    // Early out if alpha test failed / zero alpha
    if (surface.material.alpha == 0)
        discard;

    if (surface.material.normal.x != -2) // -2 signals no material normal
    {
        mat3 TBN = length(fragTBN[0]) > 0 ? fragTBN : generateTBN();
        surface.normalWS = normalize(TBN * surface.material.normal.xyz);
    }
    else
        surface.normalWS = normalize(fragTBN[2]);

    surface.NoV = saturate(dot(surface.normalWS, surface.invViewRayWS));

    vec3 color = vec3(0);

    color += evalDirectionalLight(surface);

    LightClusterInfo lightInfo =
        unpackClusterPointer(uvec2(gl_FragCoord.xy), fragZCam);

    color += evalPointLights(surface, lightInfo);

    color += evalSpotLights(surface, lightInfo);

    if (PC.ibl == 1)
        color += evalIBL(surface);

    float alpha = surface.material.alpha > 0 ? surface.material.alpha : 1.0;

    if (PC.DrawType >= DrawType_PrimitiveID)
    {
        DebugInputs di;
        di.meshID = PC.MeshID;
        di.primitiveID = gl_PrimitiveID;
        di.materialID = PC.MaterialID;
        di.position = surface.positionWS;
        di.shadingNormal = surface.normalWS;
        di.texCoord0 = fragTexCoord0;
        outColor = vec4(commonDebugDraw(PC.DrawType, di, surface.material), 1);
        return;
    }

    outColor = vec4(color, alpha);
}
