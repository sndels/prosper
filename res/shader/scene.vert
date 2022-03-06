#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "ds1_camera.glsl"

struct Transforms
{
    mat4 modelToWorld;
    mat4 normalToWorld;
};
layout(std430, set = 4, binding = 0) readonly buffer ModelInstanceTransforms
{
    Transforms instance[];
}
modelInstanceTransforms;

#include "pc_mesh.glsl"

// These need to match both vertex data and their attribute descriptions
layout(location = 0) in vec3 vertPosition;
layout(location = 1) in vec3 vertNormal;
layout(location = 2) in vec4 vertTangent;
layout(location = 3) in vec2 vertTexCoord0;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out float fragZCam;
layout(location = 2) out vec2 fragTexCoord0;
layout(location = 3) out mat3 fragTBN;

void main()
{
    Transforms trfns = modelInstanceTransforms.instance[meshPC.ModelInstanceID];

    vec4 pos = trfns.modelToWorld * vec4(vertPosition, 1.0);
    vec3 normal = normalize(mat3(trfns.normalToWorld) * vertNormal);

    // No point in generating normal basis here if no tangent is supplied
    if (length(vertTangent.xyz) > 0)
    {
        vec3 tangent = normalize(mat3(trfns.modelToWorld) * vertTangent.xyz);
        vec3 bitangent = cross(normal, tangent) * vertTangent.w;
        fragTBN = mat3(tangent, bitangent, normal);
    }
    else
        fragTBN = mat3(vec3(0), vec3(0), normal);

    fragPosition = pos.xyz / pos.w;
    fragTexCoord0 = vertTexCoord0;

    vec4 posCam = camera.worldToCamera * pos;
    fragZCam = posCam.z;

    gl_Position = camera.cameraToClip * posCam;
}
