layout(set = 1, binding = 0) uniform Camera
{
    mat4 worldToCamera;
    mat4 cameraToClip;
    vec4 eye;
    uvec2 resolution;
    float near;
    float far;
}
camera;
