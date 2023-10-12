#ifndef PROSPER_SCENE_CAMERA_HPP
#define PROSPER_SCENE_CAMERA_HPP

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Device.hpp"
#include "../gfx/RingBuffer.hpp"
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

struct PerspectiveParameters
{
    float fov{glm::radians(59.f)};
    float zN{0.1f};
    float zF{100.f};
};

// TODO:
// Split transform from other parameters as it comes from the scene graph
struct CameraParameters
{
    float fov{glm::radians(59.f)};
    float ar{16.f / 9.f};
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
    glm::vec4 eye;
    glm::uvec2 resolution;
    float near;
    float far;
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

    void lookAt(const CameraTransform &transform);
    void perspective(const PerspectiveParameters &params, float ar);
    void perspective(float ar);
    void perspective();

    // Returns true if settings changed
    bool drawUI();

    void updateBuffer(const glm::uvec2 &resolution);

    [[nodiscard]] uint32_t bufferOffset() const;
    [[nodiscard]] vk::DescriptorSetLayout descriptorSetLayout() const;
    [[nodiscard]] vk::DescriptorSet descriptorSet() const;
    [[nodiscard]] const glm::mat4 &worldToCamera() const;
    [[nodiscard]] const glm::mat4 &cameraToClip() const;
    [[nodiscard]] const CameraTransform &transform() const;
    [[nodiscard]] const CameraParameters &parameters() const;

    [[nodiscard]] static float sensorWidth() { return 0.035f; }

    void clearChangedThisFrame();
    [[nodiscard]] bool changedThisFrame() const;

    // This offset, if any, is added to internal transformation
    wheels::Optional<CameraOffset> gestureOffset;
    // Permanently applies 'offset' and empties it
    void applyGestureOffset();
    // Applies an offset without touching the held one
    void applyOffset(const CameraOffset &offset);

  private:
    void createBindingsReflection(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSet(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

    void updateWorldToCamera();

    Device *_device{nullptr};
    RingBuffer *_constantsRing{nullptr};

    CameraTransform _transform;
    CameraParameters _parameters;
    uint32_t _parametersByteOffset{0xFFFFFFFF};
    glm::mat4 _worldToCamera{1.f};
    glm::mat4 _cameraToWorld{1.f};
    glm::mat4 _cameraToClip{1.f};
    glm::mat4 _clipToWorld{1.f};

    wheels::Optional<ShaderReflection> _bindingsReflection;
    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::DescriptorSet _descriptorSet;
    bool _changedThisFrame{true};
};

#endif // PROSPER_SCENE_CAMERA_HPP
