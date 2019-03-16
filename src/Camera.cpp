#include "Camera.hpp"

using namespace glm;

Camera::Camera(Device* device, uint32_t bufferCount, const vec3& eye, const vec3& target, const vec3& up, float fov, float ar, float zN, float zF)
{
    lookAt(eye, target, up);
    perspective(fov, ar, zN, zF);
    createUniformBuffers(device, bufferCount);
}

Camera::~Camera()
{
    destroyUniformBuffers();
}

void Camera::createUniformBuffers(Device* device, uint32_t bufferCount)
{
    destroyUniformBuffers();

    _device = device;
    const vk::DeviceSize bufferSize = sizeof(CameraUniforms);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        _uniformBuffers.push_back(_device->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent
        ));
    }
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

void Camera::updateBuffer(uint32_t index)
{
    CameraUniforms uniforms;
    uniforms.worldToCamera = _worldToCamera;
    uniforms.cameraToClip = _cameraToClip;

    void* data;
    _device->logical().mapMemory(_uniformBuffers[index].memory, 0, sizeof(CameraUniforms), {}, &data);
    memcpy(data, &uniforms, sizeof(CameraUniforms));
    _device->logical().unmapMemory(_uniformBuffers[index].memory);
}

std::vector<vk::DescriptorBufferInfo> Camera::bufferInfos() const
{
    std::vector<vk::DescriptorBufferInfo> infos;
    for (auto& buffer : _uniformBuffers)
        infos.emplace_back(buffer.handle, 0, sizeof(CameraUniforms));

    return infos;
}

void Camera::destroyUniformBuffers()
{
    if (_device != nullptr) {
        for (auto& buffer : _uniformBuffers) {
            _device->logical().destroyBuffer(buffer.handle);
            _device->logical().freeMemory(buffer.memory);
        }
    }
    _uniformBuffers.clear();
}
