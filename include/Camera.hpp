#ifndef PROSPER_CAMERA_HPP
#define PROSPER_CAMERA_HPP

#include "Device.hpp"

#include <glm/glm.hpp>

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
    glm::mat4 cameraToClip;
    glm::vec4 eye;
    glm::uvec2 resolution;
    float near;
    float far;
};

class Camera
{
  public:
    Camera(
        Device *device, vk::DescriptorPool descriptorPool,
        uint32_t swapImageCount, vk::ShaderStageFlags stageFlags);
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

    void updateBuffer(uint32_t index, const glm::uvec2 &resolution);

    [[nodiscard]] std::vector<vk::DescriptorBufferInfo> bufferInfos() const;
    [[nodiscard]] const vk::DescriptorSetLayout &descriptorSetLayout() const;
    [[nodiscard]] const vk::DescriptorSet &descriptorSet(uint32_t index) const;
    [[nodiscard]] const glm::mat4 &worldToCamera() const;
    [[nodiscard]] const glm::mat4 &cameraToClip() const;
    [[nodiscard]] const CameraParameters &parameters() const;

    // This offset, if any, is added to internal transformation
    std::optional<CameraOffset> offset;
    // Permanently applies 'offset' and empties it
    void applyOffset();

  private:
    void createUniformBuffers(uint32_t swapImageCount);
    // Create uniform buffers first
    void createDescriptorSets(
        vk::DescriptorPool descriptorPool, uint32_t swapImageCount,
        vk::ShaderStageFlags stageFlags);

    void updateWorldToCamera();

    Device *_device{nullptr};
    CameraParameters _parameters;
    glm::mat4 _worldToClip{1.f};
    glm::mat4 _worldToCamera{1.f};
    glm::mat4 _cameraToClip{1.f};

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;
    std::vector<Buffer> _uniformBuffers;
};

#endif // PROSPER_CAMERA_HPP
