#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

#include "scene/camera.glsl"

layout(location = 0) in vec3 vertPosition;

layout(location = 0) out vec3 fragTexCoord;

void main()
{
    fragTexCoord = vertPosition;
    vec4 pos = camera.cameraToClip * mat4(mat3(camera.worldToCamera)) *
               vec4(vertPosition, 1);
    // Put the skybox at depth 0 in NDC since we use reverse-z
    gl_Position = vec4(pos.xy, 0, pos.w);
}
