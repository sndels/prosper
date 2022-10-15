#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"
#include "geometry.glsl"

struct Transforms
{
    mat4 modelToWorld;
    mat4 normalToWorld;
};
layout(std430, set = MODEL_INSTANCE_TRFNS_SET, binding = 0) readonly buffer
    ModelInstanceTransforms
{
    Transforms instance[];
}
modelInstanceTransforms;

#include "scene_pc.glsl"

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out float fragZCam;
layout(location = 2) out vec2 fragTexCoord0;
layout(location = 3) out mat3 fragTBN;

void main()
{
    Transforms trfns =
        modelInstanceTransforms.instance[scenePC.ModelInstanceID];
    Vertex vertex = loadVertex(scenePC.MeshID, gl_VertexIndex);

    vec4 pos = trfns.modelToWorld * vec4(vertex.Position, 1.0);
    vec3 normal = normalize(mat3(trfns.normalToWorld) * vertex.Normal);

    // No point in generating normal basis here if no tangent is supplied
    if (length(vertex.Tangent.xyz) > 0)
    {
        vec3 tangent = normalize(mat3(trfns.modelToWorld) * vertex.Tangent.xyz);
        vec3 bitangent = cross(normal, tangent) * vertex.Tangent.w;
        fragTBN = mat3(tangent, bitangent, normal);
    }
    else
        fragTBN = mat3(vec3(0), vec3(0), normal);

    fragPosition = pos.xyz / pos.w;
    fragTexCoord0 = vertex.TexCoord0;

    vec4 posCam = camera.worldToCamera * pos;
    fragZCam = posCam.z;

    gl_Position = camera.cameraToClip * posCam;
}
