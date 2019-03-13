#include "Camera.hpp"

using namespace glm;

Camera::Camera(const vec3& eye, const vec3& target, const vec3& up, float fov, float ar, float zN, float zF)
{
    lookAt(eye, target, up);
    perspective(fov, ar, zN, zF);
}

void Camera::lookAt(const vec3& eye, const vec3& target, const vec3& up)
{
    vec3 fwd = normalize(target - eye);
    orient(eye, fwd, up);
}

void Camera::orient(const vec3& eye, const vec3& fwd, const vec3& up)
{
    vec3 right = normalize(cross(up, fwd));
    vec3 newUp = normalize(cross(fwd, right));

    // Left handed camera
    _worldToCamera = mat4(         right.x,          newUp.x,          fwd.x, 0.f,
                                   right.y,          newUp.y,          fwd.y, 0.f,
                                   right.z,          newUp.z,          fwd.z, 0.f,
                          -dot(right, eye), -dot(newUp, eye), -dot(fwd, eye), 1.f);

    _worldToClip = _cameraToClip * _worldToCamera;
}

void Camera::perspective(float fov, float ar, float zN, float zF)
{
    float tf = 1.f / tanf(fov * 0.5);

    // "DX" projection matrix with flipped y to match Vulkan clip space
    _cameraToClip = mat4(tf / ar, 0.f,                  0.f, 0.f,
                             0.f, -tf,                  0.f, 0.f,
                             0.f, 0.f,       zF / (zF - zN), 1.f,
                             0.f, 0.f, -zN * zF / (zF - zN), 0.f);

    _worldToClip = _cameraToClip * _worldToCamera;
}

const mat4& Camera::worldToClip() const
{
    return _worldToClip;
}
