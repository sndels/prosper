#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "ds1_camera.glsl"

layout(set = 2, binding = 0) uniform Object { mat4 modelToWorld; }
object;

// These need to match both vertex data and their attribute descriptions
layout(location = 0) in vec3 vertPosition;
layout(location = 1) in vec3 vertNormal;
layout(location = 2) in vec4 vertTangent;
layout(location = 3) in vec2 vertTexCoord0;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec2 fragTexCoord0;
layout(location = 2) out mat3 fragTBN;

void main()
{
    vec4 pos = object.modelToWorld * vec4(vertPosition, 1.0);
    vec3 normal =
        normalize(transpose(inverse(mat3(object.modelToWorld))) * vertNormal);

    // No point in generating normal basis here if no tangent is supplied
    if (length(vertTangent.xyz) > 0)
    {
        vec3 tangent = normalize(mat3(object.modelToWorld) * vertTangent.xyz);
        vec3 bitangent = cross(normal, tangent) * vertTangent.w;
        fragTBN = mat3(tangent, bitangent, normal);
    }
    else
        fragTBN = mat3(vec3(0), vec3(0), normal);

    fragPosition = pos.xyz / pos.w;
    fragTexCoord0 = vertTexCoord0;

    gl_Position = camera.cameraToClip * camera.worldToCamera * pos;
}
