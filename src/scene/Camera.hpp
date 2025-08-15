#ifndef PROSPER_SCENE_CAMERA_HPP
#define PROSPER_SCENE_CAMERA_HPP

#include "gfx/Fwd.hpp"
#include "gfx/ShaderReflection.hpp"

#include <glm/glm.hpp>
#include <shader_structs/scene/camera.h>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

namespace scene
{

struct CameraOffset
{
    glm::vec3 eye{0.f, 0.f, 0.f};
    glm::vec3 target{0.f, 0.f, 0.f};
    bool flipUp{false};
};

struct CameraTransform
{
    glm::vec3 eye{1.f, 0.5f, 1.f};
    glm::vec3 target{0.f, 0.f, 0.f};
    glm::vec3 up{0.f, 1.f, 0.f};

    [[nodiscard]] CameraTransform apply(CameraOffset const &offset) const
    {
        return CameraTransform{
            .eye = eye + offset.eye,
            .target = target + offset.target,
            .up = offset.flipUp ? -up : up,
        };
    }
};

// TODO:
// Split transform from other parameters as it comes from the scene graph
struct CameraParameters
{
    float fov{glm::radians(59.f)};
    float zN{0.1f};
    float zF{100.f};
    float apertureDiameter{0.00001f};
    float focusDistance{1.f};
    float focalLength{0.f};
};

struct FrustumCorners
{
    glm::vec3 bottomLeftNear{0.f};
    glm::vec3 bottomRightNear{0.f};
    glm::vec3 topLeftNear{0.f};
    glm::vec3 topRightNear{0.f};
    glm::vec3 bottomLeftFar{0.f};
    glm::vec3 bottomRightFar{0.f};
    glm::vec3 topLeftFar{0.f};
    glm::vec3 topRightFar{0.f};
};

class Camera
{
  public:
    static constexpr const char *sCameraBindingName = "camera";

    Camera() noexcept = default;
    ~Camera();

    Camera(const Camera &other) = delete;
    Camera(Camera &&other) = delete;
    Camera &operator=(const Camera &other) = delete;
    Camera &operator=(Camera &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc, gfx::RingBuffer &constantsRing);
    void endFrame();

    void lookAt(const CameraTransform &transform);
    void setParameters(const CameraParameters &parameters);
    void setJitter(bool applyJitter);
    void perspective();
    void updateResolution(const glm::uvec2 &resolution);

    void updateBuffer(const wheels::Optional<FrustumCorners> &debugFrustum);

    [[nodiscard]] uint32_t bufferOffset() const;
    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;
    [[nodiscard]] vk::DescriptorSet descriptorSet() const;
    [[nodiscard]] const CameraTransform &transform() const;
    [[nodiscard]] const CameraParameters &parameters() const;
    [[nodiscard]] const glm::mat4 &clipToCamera() const;
    [[nodiscard]] const glm::uvec2 &resolution() const;

    [[nodiscard]] static float sensorWidth() { return 0.035f; }

    [[nodiscard]] bool changedThisFrame() const;

    // This offset, if any, is added to internal transformation
    wheels::Optional<CameraOffset> gestureOffset;
    // Permanently applies 'offset' and empties it
    void applyGestureOffset();
    // Applies an offset without touching the held one
    void applyOffset(const CameraOffset &offset);

    [[nodiscard]] FrustumCorners getFrustumCorners() const;

  private:
    void createBindingsReflection(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSet(wheels::ScopedScratch scopeAlloc);

    void updateWorldToCamera();
    void updateFrustumPlanes(const FrustumCorners &corners);

    bool m_initialized{false};
    gfx::RingBuffer *m_constantsRing{nullptr};

    CameraTransform m_transform;
    CameraParameters m_parameters;
    glm::uvec2 m_resolution{};
    uint32_t m_parametersByteOffset{0xFFFF'FFFF};
    glm::mat4 m_worldToCamera{1.f};
    glm::mat4 m_cameraToWorld{1.f};
    glm::mat4 m_cameraToClip{1.f};
    glm::mat4 m_clipToCamera{1.f};
    glm::mat4 m_clipToWorld{1.f};
    glm::mat4 m_previousWorldToCamera{1.f};
    glm::mat4 m_previousCameraToClip{1.f};
    glm::vec2 m_currentJitter{0.f};
    glm::vec2 m_previousJitter{0.f};
    // These are world space plane normal,distance and normals point into the
    // frustum
    glm::vec4 m_nearPlane{0.f};
    glm::vec4 m_farPlane{0.f};
    glm::vec4 m_leftPlane{0.f};
    glm::vec4 m_rightPlane{0.f};
    glm::vec4 m_topPlane{0.f};
    glm::vec4 m_bottomPlane{0.f};
    float m_maxViewScale{1.f};

    wheels::Optional<gfx::ShaderReflection> m_bindingsReflection;
    vk::DescriptorSetLayout m_descriptorSetLayout;
    vk::DescriptorSet m_descriptorSet;
    bool m_changedThisFrame{true};
    bool m_applyJitter{false};
    size_t m_jitterIndex{0};
};

} // namespace scene

#endif // PROSPER_SCENE_CAMERA_HPP
