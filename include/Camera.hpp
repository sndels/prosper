#ifndef PROSPER_CAMERA_HPP
#define PROSPER_CAMERA_HPP

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include "Device.hpp"

struct CameraOffset
{
    glm::vec3 eye{0.f, 0.f, 0.f};
    glm::vec3 target{0.f, 0.f, 0.f};
    bool flipUp{false};
};

struct CameraParameters
{
    glm::vec3 eye{1.f, 0.5f, 1.f};
    glm::vec3 target{0.f, 0.f, 0.f};
    glm::vec3 up{0.f, 1.f, 0.f};
    float fov{glm::radians(59.f)};
    float zN{0.1f};
    float zF{100.f};

    CameraParameters apply(CameraOffset const &offset) const
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
    alignas(16) glm::mat4 worldToCamera;
    alignas(16) glm::mat4 cameraToClip;
    alignas(16) glm::vec3 eye;
};

class Camera
{
  public:
    Camera(
        Device *device, const vk::DescriptorPool descriptorPool,
        const uint32_t swapImageCount, const vk::ShaderStageFlags stageFlags);
    ~Camera();

    Camera(const Camera &other) = delete;
    Camera &operator=(const Camera &other) = delete;

    void init(CameraParameters const &params);

    void lookAt(
        const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up);
    void perspective(
        const float fov, const float ar, const float zN, const float zF);
    void perspective(const float ar);

    void updateBuffer(const uint32_t index);

    std::vector<vk::DescriptorBufferInfo> bufferInfos() const;
    const vk::DescriptorSetLayout &descriptorSetLayout() const;
    const vk::DescriptorSet &descriptorSet(const uint32_t index) const;
    const glm::mat4 &worldToCamera() const;
    const glm::mat4 &cameraToClip() const;
    const CameraParameters &parameters() const;

    // This offset, if any, is added to internal transformation
    std::optional<CameraOffset> offset;
    // Permanently applies 'offset' and empties it
    void applyOffset();

  private:
    void createUniformBuffers(const uint32_t swapImageCount);
    // Create uniform buffers first
    void createDescriptorSets(
        const vk::DescriptorPool descriptorPool, const uint32_t swapImageCount,
        const vk::ShaderStageFlags stageFlags);

    void updateWorldToCamera();

    Device *_device = nullptr;
    CameraParameters _parameters;
    glm::mat4 _worldToClip;
    glm::mat4 _worldToCamera;
    glm::mat4 _cameraToClip;

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;
    std::vector<Buffer> _uniformBuffers;
};

#endif // PROSPER_CAMERA_HPP
