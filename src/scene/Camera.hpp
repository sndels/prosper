#ifndef PROSPER_SCENE_CAMERA_HPP
#define PROSPER_SCENE_CAMERA_HPP

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Device.hpp"
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

struct PerspectiveParameters
{
    float fov{glm::radians(59.f)};
    float zN{0.1f};
    float zF{100.f};
};

struct CameraParameters
{
    glm::vec3 eye{1.f, 0.5f, 1.f};
    glm::vec3 target{0.f, 0.f, 0.f};
    glm::vec3 up{0.f, 1.f, 0.f};
    float fov{glm::radians(59.f)};
    float ar{16.f / 9.f};
    float zN{0.1f};
    float zF{100.f};

    [[nodiscard]] CameraParameters apply(CameraOffset const &offset) const
    {
        return CameraParameters{
            .eye = eye + offset.eye,
            .target = target + offset.target,
            .up = offset.flipUp ? -up : up,
            .fov = fov,
        };
    }
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
    Camera(
        wheels::ScopedScratch scopeAlloc, Device *device,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~Camera();

    Camera(const Camera &other) = delete;
    Camera(Camera &&other) = delete;
    Camera &operator=(const Camera &other) = delete;
    Camera &operator=(Camera &&other) = delete;

    void init(CameraParameters const &params);

    void lookAt(
        const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up);
    void perspective(const PerspectiveParameters &params, float ar);
    void perspective(float ar);
    void perspective();

    // Returns true if settings changed
    bool drawUI();

    void updateBuffer(uint32_t index, const glm::uvec2 &resolution);

    [[nodiscard]] wheels::StaticArray<
        vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT>
    bufferInfos() const;
    [[nodiscard]] const vk::DescriptorSetLayout &descriptorSetLayout() const;
    [[nodiscard]] const vk::DescriptorSet &descriptorSet(uint32_t index) const;
    [[nodiscard]] const glm::mat4 &worldToCamera() const;
    [[nodiscard]] const glm::mat4 &cameraToClip() const;
    [[nodiscard]] const CameraParameters &parameters() const;

    [[nodiscard]] float apertureDiameter() const;
    [[nodiscard]] float focalLength() const;
    [[nodiscard]] float focusDistance() const;
    [[nodiscard]] static float sensorWidth() { return 0.035f; }

    void clearChangedThisFrame();
    [[nodiscard]] bool changedThisFrame() const;

    // This offset, if any, is added to internal transformation
    wheels::Optional<CameraOffset> offset;
    // Permanently applies 'offset' and empties it
    void applyOffset();

  private:
    void createBindingsReflection(wheels::ScopedScratch scopeAlloc);
    void createUniformBuffers();
    // Create uniform buffers first
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

    void updateWorldToCamera();

    Device *_device{nullptr};
    CameraParameters _parameters;
    float _apertureDiameter{0.00001f};
    float _focusDistance{1.f};
    float _focalLength{0.f};
    glm::mat4 _worldToCamera{1.f};
    glm::mat4 _cameraToWorld{1.f};
    glm::mat4 _cameraToClip{1.f};
    glm::mat4 _clipToWorld{1.f};

    wheels::Optional<ShaderReflection> _bindingsReflection;
    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets;
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _uniformBuffers;
    bool _changedThisFrame{true};
};

#endif // PROSPER_SCENE_CAMERA_HPP