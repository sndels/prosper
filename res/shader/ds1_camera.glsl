layout(set = 1, binding = 0) uniform Camera
{
    mat4 worldToCamera;
    mat4 cameraToClip;
    vec3 eye;
}
camera;
