#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

#include "scene/camera.glsl"

layout(std430, set = GEOMETRY_SET, binding = 0) readonly buffer GeometryBuffers
{
    float data[];
}
geometryBuffers;

layout(location = 0) out vec3 fragColor;

vec3 loadVec3(uint line, uint v)
{
    return vec3(
        geometryBuffers.data[line * 9 + 3 * v + 0],
        geometryBuffers.data[line * 9 + 3 * v + 1],
        geometryBuffers.data[line * 9 + 3 * v + 2]);
}

void main()
{
    uint line = gl_VertexIndex / 2;
    vec3 pos = loadVec3(line, gl_VertexIndex % 2);

    gl_Position = camera.cameraToClip * camera.worldToCamera * vec4(pos, 1);
    fragColor = loadVec3(line, 2);
}
