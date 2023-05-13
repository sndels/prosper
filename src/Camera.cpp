#include "Camera.hpp"

#include <iostream>

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Utils.hpp"

using namespace glm;
using namespace wheels;

Camera::Camera(Device *device, RenderResources *renderResources)
: _device{device}
, _renderResources{renderResources}
{
    assert(_device != nullptr);
    assert(_renderResources != nullptr);

    printf("Creating Camera\n");

    createUniformBuffers();
    createDescriptorSets();
}

Camera::~Camera()
{
    if (_device != nullptr)
    {
        _device->logical().destroy(_descriptorSetLayout);
        for (auto &buffer : _uniformBuffers)
            _device->destroy(buffer);

        _uniformBuffers.clear();
    }
}

void Camera::init(CameraParameters const &params)
{
    _parameters = params;

    updateWorldToCamera();
}

void Camera::lookAt(const vec3 &eye, const vec3 &target, const vec3 &up)
{
    _parameters = CameraParameters{
        .eye = eye,
        .target = target,
        .up = up,
    };

    updateWorldToCamera();
}

void Camera::perspective(const PerspectiveParameters &params, const float ar)
{
    _parameters.fov = params.fov;
    _parameters.zN = params.zN;
    _parameters.zF = params.zF;

    perspective(ar);
}

void Camera::perspective(const float ar)
{
    const auto fov = _parameters.fov;
    const auto zN = _parameters.zN;
    const auto zF = _parameters.zF;

    const float tf = 1.f / tanf(fov * 0.5f);

    // From glTF spec with flipped y and z in [0,1]

    // clang-format off
    _cameraToClip = mat4{1.f,  0.f,  0.f,  0.f,
                         0.f, -1.f,  0.f,  0.f,
                         0.f,  0.f, 0.5f,  0.f,
                         0.f, 0.f,  0.5f, 1.f } *
                    mat4{tf / ar, 0.f,                     0.f,  0.f,
                             0.f,  tf,                     0.f,  0.f,
                             0.f, 0.f,   (zF + zN) / (zN - zF), -1.f,
                             0.f, 0.f, 2 * zF * zN / (zN - zF),  0.f};
    // clang-format on

    _clipToWorld = inverse(_cameraToClip * _worldToCamera);
}

void Camera::updateBuffer(const uint32_t index, const uvec2 &resolution)
{
    if (offset.has_value())
    {
        updateWorldToCamera();
    }

    CameraUniforms uniforms{
        .worldToCamera = _worldToCamera,
        .cameraToClip = _cameraToClip,
        .clipToWorld = _clipToWorld,
        .eye =
            vec4{
                offset.has_value() ? _parameters.apply(*offset).eye
                                   : _parameters.eye,
                1.f},
        .resolution = resolution,
        .near = _parameters.zN,
        .far = _parameters.zF,
    };

    memcpy(_uniformBuffers[index].mapped, &uniforms, sizeof(CameraUniforms));
}

StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> Camera::
    bufferInfos() const
{
    StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> infos;
    for (const auto &buffer : _uniformBuffers)
        infos.push_back(vk::DescriptorBufferInfo{
            .buffer = buffer.handle,
            .offset = 0,
            .range = sizeof(CameraUniforms),
        });

    return infos;
}

const vk::DescriptorSetLayout &Camera::descriptorSetLayout() const
{
    if (!_descriptorSetLayout)
        throw std::runtime_error(
            "Camera: Called descriptorSetLayout before createDescriptorSets");
    return _descriptorSetLayout;
}

const vk::DescriptorSet &Camera::descriptorSet(const uint32_t index) const
{
    if (!_descriptorSetLayout)
        throw std::runtime_error(
            "Camera: Called descriptorSet before createDescriptorSets");
    return _descriptorSets[index];
}

const glm::mat4 &Camera::worldToCamera() const { return _worldToCamera; }

const glm::mat4 &Camera::cameraToClip() const { return _cameraToClip; }

const CameraParameters &Camera::parameters() const { return _parameters; }

void Camera::clearChangedThisFrame() { _changedThisFrame = false; }

bool Camera::changedThisFrame() const { return _changedThisFrame; }

void Camera::applyOffset()
{
    if (offset.has_value())
    {
        _parameters = _parameters.apply(*offset);
        offset.reset();
    }

    updateWorldToCamera();
}

void Camera::createUniformBuffers()
{
    const vk::DeviceSize bufferSize = sizeof(CameraUniforms);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _uniformBuffers.push_back(_device->createBuffer(BufferCreateInfo{
            .byteSize = bufferSize,
            .usage = vk::BufferUsageFlagBits::eUniformBuffer,
            .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent,
            .createMapped = true,
            .debugName = "CameraUnfiroms",
        }));
    }
}

void Camera::createDescriptorSets()
{
    const vk::DescriptorSetLayoutBinding layoutBinding{
        .binding = 0, // binding
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1, // descriptorCount
        .stageFlags = vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment |
                      vk::ShaderStageFlagBits::eCompute |
                      vk::ShaderStageFlagBits::eRaygenKHR,
    };
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &layoutBinding,
        });

    StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
    layouts.resize(MAX_FRAMES_IN_FLIGHT, _descriptorSetLayout);
    _descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    _renderResources->staticDescriptorsAlloc.allocate(layouts, _descriptorSets);

    const auto infos = bufferInfos();
    for (size_t i = 0; i < _descriptorSets.size(); ++i)
    {
        const vk::WriteDescriptorSet descriptorWrite{
            .dstSet = _descriptorSets[i],
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &infos[i],
        };
        _device->logical().updateDescriptorSets(
            1, &descriptorWrite, 0, nullptr);
    }
}

void Camera::updateWorldToCamera()
{
    const auto parameters =
        offset.has_value() ? _parameters.apply(*offset) : _parameters;
    auto const &[eye, target, up, _fov, _zN, _zF] = parameters;

    const vec3 fwd = normalize(target - eye);
    const vec3 z = -fwd;
    const vec3 right = normalize(cross(up, z));
    const vec3 newUp = normalize(cross(z, right));

    // Right handed camera
    _worldToCamera =
        mat4{right.x,          newUp.x,          z.x,          0.f,
             right.y,          newUp.y,          z.y,          0.f,
             right.z,          newUp.z,          z.z,          0.f,
             -dot(right, eye), -dot(newUp, eye), -dot(z, eye), 1.f};

    _clipToWorld = inverse(_cameraToClip * _worldToCamera);

    _changedThisFrame = true;
}
