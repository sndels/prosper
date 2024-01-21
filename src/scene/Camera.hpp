#ifndef PROSPER_SCENE_CAMERA_HPP
#define PROSPER_SCENE_CAMERA_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../utils/Utils.hpp"
#include <glm/glm.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

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

// Vector types in uniforms need to be aligned to 16 bytes
struct CameraUniforms
{
    glm::mat4 worldToCamera;
    glm::mat4 cameraToWorld;
    glm::mat4 cameraToClip;
    glm::mat4 clipToWorld;
    glm::mat4 previousWorldToCamera;
    glm::mat4 previousCameraToClip;
    glm::vec4 eye;
    glm::vec4 nearPlane;
    glm::vec4 farPlane;
    glm::vec4 leftPlane;
    glm::vec4 rightPlane;
    glm::vec4 topPlane;
    glm::vec4 bottomPlane;
    glm::uvec2 resolution;
    glm::vec2 currentJitter;
    glm::vec2 previousJitter;
    float near;
    float far;
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

    Camera(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RingBuffer *constantsRing, DescriptorAllocator *staticDescriptorsAlloc);
    ~Camera();

    Camera(const Camera &other) = delete;
    Camera(Camera &&other) = delete;
    Camera &operator=(const Camera &other) = delete;
    Camera &operator=(Camera &&other) = delete;

    void init(const CameraTransform &transform, const CameraParameters &params);
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

    [[nodiscard]] static float sensorWidth() { return 0.035f; }

    [[nodiscard]] bool changedThisFrame() const;

    // This offset, if any, is added to internal transformation
    wheels::Optional<CameraOffset> gestureOffset;
    // Permanently applies 'offset' and empties it
    void applyGestureOffset();
    // Applies an offset without touching the held one
    void applyOffset(const CameraOffset &offset);

    FrustumCorners getFrustumCorners() const;

  private:
    void createBindingsReflection(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSet(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

    void updateWorldToCamera();
    void updateFrustumPlanes(const FrustumCorners &corners);

    Device *_device{nullptr};
    RingBuffer *_constantsRing{nullptr};

    CameraTransform _transform;
    CameraParameters _parameters;
    glm::uvec2 _resolution{};
    uint32_t _parametersByteOffset{0xFFFFFFFF};
    glm::mat4 _worldToCamera{1.f};
    glm::mat4 _cameraToWorld{1.f};
    glm::mat4 _cameraToClip{1.f};
    glm::mat4 _clipToWorld{1.f};
    glm::mat4 _previousWorldToCamera{1.f};
    glm::mat4 _previousCameraToClip{1.f};
    glm::vec2 _currentJitter{0.f};
    glm::vec2 _previousJitter{0.f};
    // These are world space plane normal,distance and normals point into the
    // frustum
    glm::vec4 _nearPlane{0.f};
    glm::vec4 _farPlane{0.f};
    glm::vec4 _leftPlane{0.f};
    glm::vec4 _rightPlane{0.f};
    glm::vec4 _topPlane{0.f};
    glm::vec4 _bottomPlane{0.f};

    wheels::Optional<ShaderReflection> _bindingsReflection;
    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::DescriptorSet _descriptorSet;
    bool _changedThisFrame{true};
    bool _applyJitter{false};
    size_t _jitterIndex{0};
};

#endif // PROSPER_SCENE_CAMERA_HPP
