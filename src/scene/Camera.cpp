#include "Camera.hpp"

#include <iostream>

#include <imgui.h>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Device.hpp"
#include "../gfx/RingBuffer.hpp"
#include "../utils/Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sBindingSetIndex = 0;

// Halton base 2 for x and base 3 for y as suggested by Karis in
// High Quality Temporal Supersampling
const size_t sHaltonSampleCount = 8;
// NOLINTBEGIN(cert-err58-cpp) glm doesn't noexcept, but these won't throw
const std::array<vec2, sHaltonSampleCount> sHalton23 = {
    vec2{0.5f, 0.3333333333333333f},   vec2{0.25f, 0.6666666666666666f},
    vec2{0.75f, 0.1111111111111111f},  vec2{0.125f, 0.4444444444444444f},
    vec2{0.625f, 0.7777777777777778f}, vec2{0.375f, 0.2222222222222222f},
    vec2{0.875f, 0.5555555555555556f}, vec2{0.0625f, 0.8888888888888888f},
};
// NOLINTEND(cert-err58-cpp)

} // namespace

Camera::Camera(
    ScopedScratch scopeAlloc, Device *device, RingBuffer *constantsRing,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _constantsRing{constantsRing}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_constantsRing != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    printf("Creating Camera\n");

    createBindingsReflection(scopeAlloc.child_scope());
    createDescriptorSet(scopeAlloc.child_scope(), staticDescriptorsAlloc);
}

Camera::~Camera()
{
    if (_device != nullptr)
        _device->logical().destroy(_descriptorSetLayout);
}

void Camera::init(
    const CameraTransform &transform, const CameraParameters &params)
{
    _transform = transform;
    _parameters = params;

    updateWorldToCamera();
}

void Camera::endFrame()
{
    _changedThisFrame = false;
    _previousCameraToClip = _cameraToClip;
    _previousWorldToCamera = _worldToCamera;
    _previousJitter = _currentJitter;
    _jitterIndex = (_jitterIndex + 1) % sHaltonSampleCount;
}

void Camera::lookAt(const CameraTransform &transform)
{
    _transform = transform;

    updateWorldToCamera();
}

void Camera::setParameters(const CameraParameters &parameters)
{
    _parameters = parameters;
}

void Camera::setJitter(bool applyJitter) { _applyJitter = applyJitter; }

void Camera::perspective()
{
    const auto fov = _parameters.fov;
    const auto ar =
        static_cast<float>(_resolution.x) / static_cast<float>(_resolution.y);
    // Swap near and far for the magical properties of reverse-z
    // https://developer.nvidia.com/content/depth-precision-visualized
    const auto zN = _parameters.zF;
    const auto zF = _parameters.zN;

    const float tf = 1.f / tanf(fov * 0.5f);

    if (_applyJitter)
    {
        // Based on https://alextardif.com/TAA.html
        _currentJitter = sHalton23[_jitterIndex];
        _currentJitter *= 2.f;
        _currentJitter -= 1.f;
        _currentJitter /= vec2{
            static_cast<float>(_resolution.x),
            static_cast<float>(_resolution.y)};
    }
    else
        _currentJitter = vec2{0.f, 0.f};

    // From glTF spec with flipped y and z in [0,1]
    // Compensate for the flipped y projection by flipping jitter x in the
    // matrix. That way, the shader can unjitter using the original jitter value

    // clang-format off
    _cameraToClip = mat4{1.f,  0.f,  0.f,  0.f,
                         0.f, -1.f,  0.f,  0.f,
                         0.f,  0.f, 0.5f,  0.f,
                         0.f, 0.f,  0.5f, 1.f } *
                    mat4{              tf / ar,              0.f,                     0.f,  0.f,
                                           0.f,               tf,                     0.f,  0.f,
                             -_currentJitter.x, _currentJitter.y,   (zF + zN) / (zN - zF), -1.f,
                                           0.f,              0.f, 2 * zF * zN / (zN - zF),  0.f};
    // clang-format on

    _clipToWorld = inverse(_cameraToClip * _worldToCamera);

    const float sensorHeight = sensorWidth() / ar;

    _parameters.focalLength = sensorHeight * tf * 0.5f;
}

void Camera::updateResolution(const uvec2 &resolution)
{
    _resolution = resolution;
}

void Camera::updateBuffer()
{
    if (gestureOffset.has_value())
    {
        updateWorldToCamera();
    }

    // Always update perspective to have correct jitter regardless of settings
    perspective();

    const CameraUniforms uniforms{
        .worldToCamera = _worldToCamera,
        .cameraToWorld = _cameraToWorld,
        .cameraToClip = _cameraToClip,
        .clipToWorld = _clipToWorld,
        .previousWorldToCamera = _previousWorldToCamera,
        .previousCameraToClip = _previousCameraToClip,
        .eye =
            vec4{
                gestureOffset.has_value() ? _transform.apply(*gestureOffset).eye
                                          : _transform.eye,
                1.f},
        .resolution = _resolution,
        .currentJitter = _currentJitter,
        .previousJitter = _previousJitter,
        .near = _parameters.zN,
        .far = _parameters.zF,
    };
    _parametersByteOffset = _constantsRing->write_value(uniforms);
}

uint32_t Camera::bufferOffset() const { return _parametersByteOffset; }

vk::DescriptorSetLayout Camera::descriptorSetLayout() const
{
    return _descriptorSetLayout;
}

vk::DescriptorSet Camera::descriptorSet() const { return _descriptorSet; }

const CameraTransform &Camera::transform() const { return _transform; }

const CameraParameters &Camera::parameters() const { return _parameters; }

bool Camera::changedThisFrame() const { return _changedThisFrame; }

void Camera::applyGestureOffset()
{
    if (gestureOffset.has_value())
    {
        _transform = _transform.apply(*gestureOffset);
        gestureOffset.reset();
    }

    updateWorldToCamera();
}

void Camera::applyOffset(const CameraOffset &offset)
{
    _transform = _transform.apply(offset);

    updateWorldToCamera();
}

void Camera::createBindingsReflection(ScopedScratch scopeAlloc)
{
    const size_t len = 32;
    String defines{scopeAlloc, len};
    appendDefineStr(defines, "CAMERA_SET", sBindingSetIndex);
    WHEELS_ASSERT(defines.size() <= len);

    Optional<ShaderReflection> compResult = _device->reflectShader(
        scopeAlloc.child_scope(),
        Device::CompileShaderModuleArgs{
            .relPath = "shader/scene/camera.glsl",
            .defines = defines,
        },
        true);
    if (!compResult.has_value())
        throw std::runtime_error("Failed to create camera bindings reflection");

    _bindingsReflection = WHEELS_MOV(*compResult);
}

void Camera::createDescriptorSet(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(_bindingsReflection.has_value());
    _descriptorSetLayout = _bindingsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), *_device, 0,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eRaygenKHR);

    _descriptorSet = staticDescriptorsAlloc->allocate(_descriptorSetLayout);

    const StaticArray descriptorInfos{
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = _constantsRing->buffer(),
            .range = sizeof(CameraUniforms),
        }},
    };
    const Array descriptorWrites =
        _bindingsReflection->generateDescriptorWrites(
            scopeAlloc, sBindingSetIndex, _descriptorSet, descriptorInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void Camera::updateWorldToCamera()
{
    const auto transform = gestureOffset.has_value()
                               ? _transform.apply(*gestureOffset)
                               : _transform;
    auto const &[eye, target, up] = transform;

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
    _cameraToWorld = inverse(_worldToCamera);

    _clipToWorld = inverse(_cameraToClip * _worldToCamera);

    _changedThisFrame = true;
}
