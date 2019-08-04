#include "Camera.hpp"

#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

Camera::~Camera()
{
    if (_device) {
        _device->logical().destroy(_descriptorSetLayout);
        destroyUniformBuffers();
    }
}

void Camera::createUniformBuffers(std::shared_ptr<Device> device, const uint32_t swapImageCount)
{
    destroyUniformBuffers();

    _device = device;
    const vk::DeviceSize bufferSize = sizeof(CameraUniforms);

    for (uint32_t i = 0; i < swapImageCount; ++i) {
        _uniformBuffers.push_back(_device->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        ));
    }
}

void Camera::createDescriptorSets(const vk::DescriptorPool descriptorPool, const uint32_t swapImageCount, const vk::ShaderStageFlags stageFlags)
{
    const vk::DescriptorSetLayoutBinding layoutBinding{
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        stageFlags
    };
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &layoutBinding
    });

    const std::vector<vk::DescriptorSetLayout> layouts(
        swapImageCount,
        _descriptorSetLayout
    );
    _descriptorSets = _device->logical().allocateDescriptorSets({
        descriptorPool,
        static_cast<uint32_t>(layouts.size()),
        layouts.data()
    });

    const auto infos = bufferInfos();
    for (size_t i = 0; i < _descriptorSets.size(); ++i) {
        const vk::WriteDescriptorSet descriptorWrite{
            _descriptorSets[i],
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            vk::DescriptorType::eUniformBuffer,
            nullptr, // pImageInfo
            &infos[i]
        };
        _device->logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }
}

void Camera::lookAt(const vec3& eye, const vec3& target, const vec3& up)
{
    vec3 fwd = normalize(target - eye);
    orient(eye, fwd, up);
}

void Camera::orient(const vec3& eye, const vec3& fwd, const vec3& up)
{
    _eye = eye;
    vec3 z = -fwd;
    vec3 right = normalize(cross(up, z));
    vec3 newUp = normalize(cross(z, right));

    // Right handed camera
    _worldToCamera = mat4{         right.x,          newUp.x,          z.x, 0.f,
                                   right.y,          newUp.y,          z.y, 0.f,
                                   right.z,          newUp.z,          z.z, 0.f,
                          -dot(right, eye), -dot(newUp, eye), -dot(z, eye), 1.f};

    _worldToClip = _cameraToClip * _worldToCamera;
}

void Camera::perspective(const float fov, const float ar, const float zN, const float zF)
{
    const float tf = 1.f / tanf(fov * 0.5);

    // From glTF spec with flipped y and z in [0,1]
    _cameraToClip = mat4{1.f,  0.f,  0.f, 0.f,
                         0.f, -1.f,  0.f, 0.f,
                         0.f,  0.f, 0.5f, 0.f,
                         0.f,  0.f, 0.5f, 1.f} *
                    mat4{tf / ar, 0.f,                         0.f,  0.f,
                             0.f,  tf,                         0.f,  0.f,
                             0.f, 0.f,       (zF + zN) / (zN - zF), -1.f,
                             0.f, 0.f,     2 * zF * zN / (zN - zF),  0.f};

    _worldToClip = _cameraToClip * _worldToCamera;
}

void Camera::orbit(const vec2& currentPos, const vec2& lastPos, const vec2& screenCenter)
{
    const auto& right = vec3{row(_worldToCamera, 0)};
    const vec2 delta = -(lastPos - currentPos) / screenCenter;

    _worldToCamera =
        _worldToCamera *
        glm::rotate(mat4{1.f}, delta.y, right) *
        glm::rotate(mat4{1.f}, delta.x, vec3{0.f, 1.f, 0.f});
    _eye = vec3{inverse(_worldToCamera)[3]};
}

void Camera::scaleOrbit(const float currentY, const float lastY, const float screenCenterY)
{
    // Move along the z-axis
    const float delta = -(currentY - lastY) / screenCenterY;
    _eye += vec3{row(_worldToCamera, 2)} * delta;
    _worldToCamera[3] = vec4{
        -dot(vec3{row(_worldToCamera, 0)}, _eye),
        -dot(vec3{row(_worldToCamera, 1)}, _eye),
        -dot(vec3{row(_worldToCamera, 2)}, _eye),
        1.f
    };
}

void Camera::updateBuffer(const uint32_t index) const
{
    CameraUniforms uniforms;
    uniforms.eye = _eye;
    uniforms.worldToCamera = _worldToCamera;
    uniforms.cameraToClip = _cameraToClip;

    void* data;
    _device->map(_uniformBuffers[index].allocation, &data);
    memcpy(data, &uniforms, sizeof(CameraUniforms));
    _device->unmap(_uniformBuffers[index].allocation);
}

std::vector<vk::DescriptorBufferInfo> Camera::bufferInfos() const
{
    std::vector<vk::DescriptorBufferInfo> infos;
    for (auto& buffer : _uniformBuffers)
        infos.emplace_back(buffer.handle, 0, sizeof(CameraUniforms));

    return infos;
}

const vk::DescriptorSetLayout& Camera::descriptorSetLayout() const
{
    if (!_descriptorSetLayout)
        throw std::runtime_error("Camera: Called descriptorSetLayout before createDescriptorSets");
    return _descriptorSetLayout;
}

const vk::DescriptorSet& Camera::descriptorSet(const uint32_t index) const
{
    if (!_descriptorSetLayout)
        throw std::runtime_error("Camera: Called descriptorSet before createDescriptorSets");
    return _descriptorSets[index];
}

const glm::mat4& Camera::worldToCamera() const
{
    return _worldToCamera;
}

const glm::mat4& Camera::cameraToClip() const
{
    return _cameraToClip;
}

void Camera::destroyUniformBuffers()
{
    if (_device != nullptr) {
        for (auto& buffer : _uniformBuffers)
            _device->destroy(buffer);
    }
    _uniformBuffers.clear();
}
