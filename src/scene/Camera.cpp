#include "Camera.hpp"

#include <iostream>

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include "../utils/Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const uint32_t sBindingSetIndex = 0;

}

Camera::Camera(
    ScopedScratch scopeAlloc, Device *device, RingBuffer *constantsRing,
    DescriptorAllocator *staticDescriptorsAlloc)
: _device{device}
, _constantsRing{constantsRing}
{
    assert(_device != nullptr);
    assert(_constantsRing != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating Camera\n");

    createBindingsReflection(scopeAlloc.child_scope());
    createDescriptorSet(scopeAlloc.child_scope(), staticDescriptorsAlloc);
}

Camera::~Camera()
{
    if (_device != nullptr)
        _device->logical().destroy(_descriptorSetLayout);
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
    _parameters.ar = ar;
    _parameters.zN = params.zN;
    _parameters.zF = params.zF;

    perspective();
}

void Camera::perspective(const float ar)
{
    _parameters.ar = ar;

    perspective();
}

void Camera::perspective()
{
    const auto fov = _parameters.fov;
    const auto ar = _parameters.ar;
    // Swap near and far for the magical properties of reverse-z
    // https://developer.nvidia.com/content/depth-precision-visualized
    const auto zN = _parameters.zF;
    const auto zF = _parameters.zN;

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

    const float sensorHeight = sensorWidth() / ar;

    _focalLength = sensorHeight * tf * 0.5f;
}

bool Camera::drawUI()
{
    bool changed = false;

    ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // TODO: Tweak this in millimeters?
    changed |= ImGui::DragFloat(
        "Aperture Diameter", &_apertureDiameter, 0.00001f, 0.0000001f, 0.1f,
        "%.6f");
    changed |= ImGui::DragFloat(
        "FocusDistance", &_focusDistance, 0.01f, 0.001f, 100.f);

    float fovDegrees = degrees(_parameters.fov);
    if (ImGui::DragFloat("Field of View", &fovDegrees, 0.1f, 0.1f, 179.f))
    {
        _parameters.fov = radians(fovDegrees);
        perspective();
        changed = true;
    }

    ImGui::Text("Focal length: %.3fmm", _focalLength * 1e3);

    ImGui::End();

    return changed;
}

void Camera::updateBuffer(const uvec2 &resolution)
{
    if (gestureOffset.has_value())
    {
        updateWorldToCamera();
    }

    const CameraUniforms uniforms{
        .worldToCamera = _worldToCamera,
        .cameraToWorld = _cameraToWorld,
        .cameraToClip = _cameraToClip,
        .clipToWorld = _clipToWorld,
        .eye =
            vec4{
                gestureOffset.has_value()
                    ? _parameters.apply(*gestureOffset).eye
                    : _parameters.eye,
                1.f},
        .resolution = resolution,
        .near = _parameters.zN,
        .far = _parameters.zF,
    };
    _parametersByteOffset = _constantsRing->write(uniforms);
}

uint32_t Camera::bufferOffset() const { return _parametersByteOffset; }

vk::DescriptorSetLayout Camera::descriptorSetLayout() const
{
    return _descriptorSetLayout;
}

vk::DescriptorSet Camera::descriptorSet() const { return _descriptorSet; }

const glm::mat4 &Camera::worldToCamera() const { return _worldToCamera; }

const glm::mat4 &Camera::cameraToClip() const { return _cameraToClip; }

const CameraParameters &Camera::parameters() const { return _parameters; }

float Camera::apertureDiameter() const { return _apertureDiameter; }

float Camera::focalLength() const { return _focalLength; }

float Camera::focusDistance() const { return _focusDistance; }

void Camera::clearChangedThisFrame() { _changedThisFrame = false; }

bool Camera::changedThisFrame() const { return _changedThisFrame; }

void Camera::applyGestureOffset()
{
    if (gestureOffset.has_value())
    {
        _parameters = _parameters.apply(*gestureOffset);
        gestureOffset.reset();
    }

    updateWorldToCamera();
}

void Camera::applyOffset(const CameraOffset &offset)
{
    _parameters = _parameters.apply(offset);

    updateWorldToCamera();
}

void Camera::createBindingsReflection(ScopedScratch scopeAlloc)
{
    String defines{scopeAlloc, 64};
    appendDefineStr(defines, "CAMERA_SET", sBindingSetIndex);

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
    assert(_bindingsReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _bindingsReflection->generateLayoutBindings(
            scopeAlloc, 0,
            vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eCompute |
                vk::ShaderStageFlagBits::eRaygenKHR);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    _descriptorSet = staticDescriptorsAlloc->allocate(_descriptorSetLayout);

    const StaticArray descriptorWrites =
        _bindingsReflection->generateDescriptorWrites<1>(
            sBindingSetIndex, _descriptorSet,
            StaticArray{{
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _constantsRing->buffer(),
                    .range = sizeof(CameraUniforms),
                }},
            }});

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void Camera::updateWorldToCamera()
{
    const auto parameters = gestureOffset.has_value()
                                ? _parameters.apply(*gestureOffset)
                                : _parameters;
    auto const &[eye, target, up, _fov, _ar, _zN, _zF] = parameters;

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
