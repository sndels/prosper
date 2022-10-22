#ifndef RAY_GLSL
#define RAY_GLSL

struct Ray
{
    vec3 o;
    vec3 d;
    float tMin;
    float tMax;
};

// Ported from RT Gems 2 chapter 14
// This doesn't use the inverse projection matrix because of precision issues
// caused by near and far
Ray pinholeCameraRay(vec2 uv)
{
    vec2 nd = uv * 2 - 1;

    // Set up the ray.
    Ray ray;
    ray.o = camera.eye.xyz;
    ray.tMin = 0.0;
    ray.tMax = 1.0 / 0.0; // INF

    // Extract the aspect ratio and fov from the projection matrix.
    float aspect = camera.cameraToClip[1][1] / camera.cameraToClip[0][0];
    float tanHalfFovY = 1.0 / camera.cameraToClip[1][1];

    vec3 right = vec3(
        camera.worldToCamera[0].x, camera.worldToCamera[1].x,
        camera.worldToCamera[2].x);
    vec3 up = vec3(
        camera.worldToCamera[0].y, camera.worldToCamera[1].y,
        camera.worldToCamera[2].y);
    vec3 fwd = -vec3(
        camera.worldToCamera[0].z, camera.worldToCamera[1].z,
        camera.worldToCamera[2].z);

    // Compute the ray direction.
    ray.d = normalize(
        (nd.x * right * tanHalfFovY * aspect) + (nd.y * up * tanHalfFovY) +
        fwd);

    return ray;
}

#endif // RAY_GLSL
