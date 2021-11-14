#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform Camera
{
    mat4 worldToClip; // This shouldn't have camera translation to keep skybox
                      // stationary
}
camera;

layout(location = 0) in vec3 vertPosition;

layout(location = 0) out vec3 fragTexCoord;

void main()
{
    fragTexCoord = vertPosition;
    vec4 pos = camera.worldToClip * vec4(vertPosition, 1);
    // Make sure the skybox is at at depth 1 in NDC
    gl_Position = pos.xyww;
}
