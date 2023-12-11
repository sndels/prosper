#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#ifdef USE_GBUFFER_PC
#include "gbuffer_pc.glsl"
#else // !USE_GBUFFER_PC
#include "forward_pc.glsl"
#endif // USE_GBUFFER_PC

#include "scene/camera.glsl"
#include "scene/geometry.glsl"
#include "scene/transforms.glsl"

layout(location = 0) out vec3 fragPositionWorld;
layout(location = 1) out float fragZCam;
layout(location = 2) out vec2 fragTexCoord0;
layout(location = 3) out vec4 fragPositionNDC;
layout(location = 4) out vec4 fragPrevPositionNDC;
layout(location = 5) out mat3 fragTBN;

void main()
{
    Transforms trfn = modelInstanceTransforms.instance[PC.ModelInstanceID];
    Vertex vertexModel = loadVertex(PC.MeshID, gl_VertexIndex);
    Vertex vertexWorld = transform(vertexModel, trfn);

    if (vertexWorld.Tangent.w != 0)
        fragTBN = generateTBN(vertexWorld.Normal, vertexWorld.Tangent);
    else
        fragTBN = mat3(vec3(0), vec3(0), vertexWorld.Normal);

    fragPositionWorld = vertexWorld.Position;
    fragTexCoord0 = vertexWorld.TexCoord0;

    vec4 posCam = camera.worldToCamera * vec4(vertexWorld.Position, 1);
    fragZCam = posCam.z;

    vec4 posNDC = camera.cameraToClip * posCam;
    fragPositionNDC = posNDC;

    Transforms prevTrfn = trfn;
    if (PC.previousTransformValid == 1)
        prevTrfn = previousModelInstanceTransforms.instance[PC.ModelInstanceID];
    vec3 prevPositionWorld = worldPosition(vertexModel, prevTrfn);

    fragPrevPositionNDC = camera.previousCameraToClip *
                          camera.previousWorldToCamera *
                          vec4(prevPositionWorld, 1.);

    gl_Position = posNDC;
}
