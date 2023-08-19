#ifndef RT_RAY_GLSL
#define RT_RAY_GLSL

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
    vec3 fwd = cameraWorldFwd();

    // Compute the ray direction.
    ray.d = normalize(
        (nd.x * right * tanHalfFovY * aspect) + (nd.y * up * tanHalfFovY) +
        fwd);

    return ray;
}

// Adapted from RT Gems 2 chapter 3
Ray thinLensCameraRay(
    vec2 uv, vec2 lensOffset, float apertureDiameter, float focusDistance,
    float focalLength)
{
    Ray pinholeRay = pinholeCameraRay(uv);

    float theta = lensOffset.x * 2. * PI;
    float radius = lensOffset.y;

    float u = cos(theta) * sqrt(radius);
    float v = sin(theta) * sqrt(radius);

    // Original calculated focal plane from image distance and had the ray in
    // view space. Let's use the focus distance we already have and keep this in
    // world space.
    vec3 focusPoint =
        pinholeRay.o +
        pinholeRay.d * (focusDistance / dot(pinholeRay.d, cameraWorldFwd()));

    float fStop = focalLength / apertureDiameter;
    float circleOfConfusionRadius = focalLength / (2. * fStop);

    vec3 lensPos = vec3(1, 0, 0) * (u * circleOfConfusionRadius) +
                   vec3(0, 1, 0) * (v * circleOfConfusionRadius);

    Ray ray;
    ray.o = (camera.cameraToWorld * vec4(lensPos, 1)).xyz;
    ray.d = normalize(focusPoint - ray.o);
    ray.tMin = 0.;
    ray.tMax = 1. / 0.; // INF

    return ray;
}

#endif // RT_RAY_GLSL
