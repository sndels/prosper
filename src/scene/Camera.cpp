#include "Camera.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "gfx/RingBuffer.hpp"
#include "utils/Logger.hpp"
#include "utils/Utils.hpp"

#include <glm/gtc/matrix_access.hpp>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace scene
{

namespace
{

const uint32_t sBindingSetIndex = 0;

// Halton base 2 for x and base 3 for y as suggested by Karis in
// High Quality Temporal Supersampling
const size_t sHaltonSampleCount = 8;
// NOLINTBEGIN(cert-err58-cpp) glm doesn't noexcept, but these won't throw
const StaticArray<vec2, sHaltonSampleCount> sHalton23{{
    vec2{0.5f, 0.3333333333333333f},
    vec2{0.25f, 0.6666666666666666f},
    vec2{0.75f, 0.1111111111111111f},
    vec2{0.125f, 0.4444444444444444f},
    vec2{0.625f, 0.7777777777777778f},
    vec2{0.375f, 0.2222222222222222f},
    vec2{0.875f, 0.5555555555555556f},
    vec2{0.0625f, 0.8888888888888888f},
}};
// NOLINTEND(cert-err58-cpp)

vec4 getPlane(vec3 p0, vec3 p1, vec3 p2)
{
    const vec3 normal = normalize(cross(p1 - p0, p2 - p0));
    const float distance = -dot(normal, p0);

    return vec4{normal, distance};
}

} // namespace

Camera::~Camera()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gfx::gDevice.logical().destroy(m_descriptorSetLayout);
}

void Camera::init(
    wheels::ScopedScratch scopeAlloc, gfx::RingBuffer &constantsRing)
{
    WHEELS_ASSERT(!m_initialized);

    m_constantsRing = &constantsRing;

    LOG_INFO("Creating Camera");

    createBindingsReflection(scopeAlloc.child_scope());
    createDescriptorSet(scopeAlloc.child_scope());

    m_initialized = true;
}

void Camera::endFrame()
{
    WHEELS_ASSERT(m_initialized);

    m_changedThisFrame = false;
    m_previousCameraToClip = m_cameraToClip;
    m_previousWorldToCamera = m_worldToCamera;
    m_previousJitter = m_currentJitter;
    m_jitterIndex = (m_jitterIndex + 1) % sHaltonSampleCount;
}

void Camera::lookAt(const CameraTransform &transform)
{
    WHEELS_ASSERT(m_initialized);

    m_transform = transform;

    updateWorldToCamera();
}

void Camera::setParameters(const CameraParameters &parameters)
{
    WHEELS_ASSERT(m_initialized);

    m_parameters = parameters;
}

void Camera::setJitter(bool applyJitter)
{
    WHEELS_ASSERT(m_initialized);

    m_applyJitter = applyJitter;
}

void Camera::perspective()
{
    WHEELS_ASSERT(m_initialized);

    const auto fov = m_parameters.fov;
    const auto ar =
        static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y);
    // Swap near and far for the magical properties of reverse-z
    // https://developer.nvidia.com/content/depth-precision-visualized
    const auto zN = m_parameters.zF;
    const auto zF = m_parameters.zN;

    const float tf = 1.f / tanf(fov * 0.5f);

    if (m_applyJitter)
    {
        // Based on https://alextardif.com/TAA.html
        m_currentJitter = sHalton23[m_jitterIndex];
        m_currentJitter *= 2.f;
        m_currentJitter -= 1.f;
        m_currentJitter /= vec2{
            static_cast<float>(m_resolution.x),
            static_cast<float>(m_resolution.y)};
    }
    else
        m_currentJitter = vec2{0.f, 0.f};

    // From glTF spec with flipped y and z in [0,1]
    // Compensate for the flipped y projection by flipping jitter x in the
    // matrix. That way, the shader can unjitter using the original jitter value

    // clang-format off
    m_cameraToClip = mat4{1.f,  0.f,  0.f,  0.f,
                         0.f, -1.f,  0.f,  0.f,
                         0.f,  0.f, 0.5f,  0.f,
                         0.f, 0.f,  0.5f, 1.f } *
                    mat4{              tf / ar,              0.f,                     0.f,  0.f,
                                           0.f,               tf,                     0.f,  0.f,
                             -m_currentJitter.x, m_currentJitter.y,   (zF + zN) / (zN - zF), -1.f,
                                           0.f,              0.f, 2 * zF * zN / (zN - zF),  0.f};
    // clang-format on

    m_clipToCamera = inverse(m_cameraToClip);
    m_clipToWorld = inverse(m_cameraToClip * m_worldToCamera);

    const float sensorHeight = sensorWidth() / ar;

    m_parameters.focalLength = sensorHeight * tf * 0.5f;
}

void Camera::updateResolution(const uvec2 &resolution)
{
    WHEELS_ASSERT(m_initialized);

    m_resolution = resolution;
}

void Camera::updateBuffer(const wheels::Optional<FrustumCorners> &debugFrustum)
{
    WHEELS_ASSERT(m_initialized);

    if (gestureOffset.has_value())
    {
        updateWorldToCamera();
    }

    // Always update perspective to have correct jitter regardless of settings
    perspective();

    updateFrustumPlanes(
        debugFrustum.has_value() ? *debugFrustum : getFrustumCorners());

    const shader_structs::CameraUniforms uniforms{
        .worldToCamera = m_worldToCamera,
        .cameraToWorld = m_cameraToWorld,
        .cameraToClip = m_cameraToClip,
        .clipToWorld = m_clipToWorld,
        .previousWorldToCamera = m_previousWorldToCamera,
        .previousCameraToClip = m_previousCameraToClip,
        .eye =
            vec4{
                gestureOffset.has_value()
                    ? m_transform.apply(*gestureOffset).eye
                    : m_transform.eye,
                1.f},
        .nearPlane = m_nearPlane,
        .farPlane = m_farPlane,
        .leftPlane = m_leftPlane,
        .rightPlane = m_rightPlane,
        .topPlane = m_topPlane,
        .bottomPlane = m_bottomPlane,
        .resolution = m_resolution,
        .currentJitter = m_currentJitter,
        .previousJitter = m_previousJitter,
        .near = m_parameters.zN,
        .far = m_parameters.zF,
        .maxViewScale = m_maxViewScale,
    };
    m_parametersByteOffset = m_constantsRing->write_value(uniforms);
}

uint32_t Camera::bufferOffset() const
{
    WHEELS_ASSERT(m_initialized);

    return m_parametersByteOffset;
}

vk::DescriptorSetLayout Camera::descriptorSetLayout() const
{
    WHEELS_ASSERT(m_initialized);

    return m_descriptorSetLayout;
}

vk::DescriptorSet Camera::descriptorSet() const
{
    WHEELS_ASSERT(m_initialized);

    return m_descriptorSet;
}

const CameraTransform &Camera::transform() const
{
    WHEELS_ASSERT(m_initialized);

    return m_transform;
}

const CameraParameters &Camera::parameters() const
{
    WHEELS_ASSERT(m_initialized);

    return m_parameters;
}

const mat4 &Camera::clipToCamera() const { return m_clipToCamera; }

const uvec2 &Camera::resolution() const { return m_resolution; }

bool Camera::changedThisFrame() const
{
    WHEELS_ASSERT(m_initialized);

    return m_changedThisFrame;
}

void Camera::applyGestureOffset()
{
    WHEELS_ASSERT(m_initialized);

    if (gestureOffset.has_value())
    {
        m_transform = m_transform.apply(*gestureOffset);
        gestureOffset.reset();
    }

    updateWorldToCamera();
}

void Camera::applyOffset(const CameraOffset &offset)
{
    WHEELS_ASSERT(m_initialized);

    m_transform = m_transform.apply(offset);

    updateWorldToCamera();
}

FrustumCorners Camera::getFrustumCorners() const
{
    WHEELS_ASSERT(m_initialized);

    const CameraTransform transform = gestureOffset.has_value()
                                          ? m_transform.apply(*gestureOffset)
                                          : m_transform;

    const vec3 right = vec3{row(m_worldToCamera, 0)};
    const vec3 up = vec3{row(m_worldToCamera, 1)};
    // Flip so that fwd is the real camera direction in world space
    // These vectors aren't used to construct a coordinate frame so right is
    // *not* flipped for handedness correction
    const vec3 fwd = -vec3{row(m_worldToCamera, 2)};

    const float ar =
        static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y);
    const float halfYFar = m_parameters.zF * tanf(m_parameters.fov * 0.5f);
    const float halfXFar = halfYFar * ar;
    const float halfYNear = m_parameters.zN * tanf(m_parameters.fov * 0.5f);
    const float halfXNear = halfYNear * ar;

    const FrustumCorners ret{
        .bottomLeftNear = transform.eye + m_parameters.zN * fwd -
                          halfXNear * right - halfYNear * up,
        .bottomRightNear = transform.eye + m_parameters.zN * fwd +
                           halfXNear * right - halfYNear * up,
        .topLeftNear = transform.eye + m_parameters.zN * fwd -
                       halfXNear * right + halfYNear * up,
        .topRightNear = transform.eye + m_parameters.zN * fwd +
                        halfXNear * right + halfYNear * up,
        .bottomLeftFar = transform.eye + m_parameters.zF * fwd -
                         halfXFar * right - halfYFar * up,
        .bottomRightFar = transform.eye + m_parameters.zF * fwd +
                          halfXFar * right - halfYFar * up,
        .topLeftFar = transform.eye + m_parameters.zF * fwd - halfXFar * right +
                      halfYFar * up,
        .topRightFar = transform.eye + m_parameters.zF * fwd +
                       halfXFar * right + halfYFar * up,
    };

    return ret;
}

void Camera::createBindingsReflection(ScopedScratch scopeAlloc)
{
    const size_t len = 32;
    String defines{scopeAlloc, len};
    appendDefineStr(defines, "CAMERA_SET", sBindingSetIndex);
    WHEELS_ASSERT(defines.size() <= len);

    Optional<gfx::ShaderReflection> compResult = gfx::gDevice.reflectShader(
        scopeAlloc.child_scope(),
        gfx::Device::CompileShaderModuleArgs{
            .relPath = "shader/scene/camera.glsl",
            .defines = defines,
        },
        true);
    if (!compResult.has_value())
        throw std::runtime_error("Failed to create camera bindings reflection");

    m_bindingsReflection = WHEELS_MOV(*compResult);
}

void Camera::createDescriptorSet(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_bindingsReflection.has_value());
    m_descriptorSetLayout = m_bindingsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), 0,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eMeshEXT);

    m_descriptorSet =
        gfx::gStaticDescriptorsAlloc.allocate(m_descriptorSetLayout, "Camera");

    const StaticArray descriptorInfos{
        gfx::DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = m_constantsRing->buffer(),
            .range = sizeof(shader_structs::CameraUniforms),
        }},
    };
    const Array descriptorWrites =
        m_bindingsReflection->generateDescriptorWrites(
            scopeAlloc, sBindingSetIndex, m_descriptorSet, descriptorInfos);

    gfx::gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void Camera::updateWorldToCamera()
{
    const auto transform = gestureOffset.has_value()
                               ? m_transform.apply(*gestureOffset)
                               : m_transform;
    auto const &[eye, target, up] = transform;

    const vec3 fwd = normalize(target - eye);
    const vec3 z = -fwd;
    const vec3 right = normalize(cross(up, z));
    const vec3 newUp = normalize(cross(z, right));

    // Right handed camera
    m_worldToCamera =
        mat4{right.x,          newUp.x,          z.x,          0.f,
             right.y,          newUp.y,          z.y,          0.f,
             right.z,          newUp.z,          z.z,          0.f,
             -dot(right, eye), -dot(newUp, eye), -dot(z, eye), 1.f};
    m_cameraToWorld = inverse(m_worldToCamera);

    const vec3 scale{
        length(column(m_worldToCamera, 0)), length(column(m_worldToCamera, 1)),
        length(column(m_worldToCamera, 2))};
    m_maxViewScale = max(max(scale.x, scale.y), scale.z);

    m_clipToCamera = inverse(m_cameraToClip);
    m_clipToWorld = inverse(m_cameraToClip * m_worldToCamera);

    m_changedThisFrame = true;
}

void Camera::updateFrustumPlanes(const FrustumCorners &corners)
{
    // Use corners instead of shortcutting with fwd and near/far to make this
    // work with cached corners as well
    m_nearPlane = getPlane(
        corners.bottomRightNear, corners.bottomLeftNear, corners.topRightNear);
    m_farPlane = getPlane(
        corners.bottomRightFar, corners.topRightFar, corners.bottomLeftFar);
    m_leftPlane = getPlane(
        corners.bottomLeftNear, corners.bottomLeftFar, corners.topLeftNear);
    m_rightPlane = getPlane(
        corners.bottomRightNear, corners.topRightNear, corners.bottomRightFar);
    m_topPlane =
        getPlane(corners.topLeftNear, corners.topLeftFar, corners.topRightNear);
    m_bottomPlane = getPlane(
        corners.bottomLeftNear, corners.bottomRightNear, corners.bottomLeftFar);
}

} // namespace scene
