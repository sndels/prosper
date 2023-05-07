#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "forward_pc.glsl"
#include "scene/camera.glsl"
#include "scene/geometry.glsl"
#include "scene/transforms.glsl"

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out float fragZCam;
layout(location = 2) out vec2 fragTexCoord0;
layout(location = 3) out mat3 fragTBN;

void main()
{
    Vertex vertex = transform(
        loadVertex(forwardPC.MeshID, gl_VertexIndex),
        modelInstanceTransforms.instance[forwardPC.ModelInstanceID]);

    if (vertex.Tangent.w != 0)
        fragTBN = generateTBN(vertex.Normal, vertex.Tangent);
    else
        fragTBN = mat3(vec3(0), vec3(0), vertex.Normal);

    fragPosition = vertex.Position;
    fragTexCoord0 = vertex.TexCoord0;

    vec4 posCam = camera.worldToCamera * vec4(vertex.Position, 1);
    fragZCam = posCam.z;

    gl_Position = camera.cameraToClip * posCam;
}
