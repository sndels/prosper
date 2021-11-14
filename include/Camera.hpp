#ifndef PROSPER_CAMERA_HPP
#define PROSPER_CAMERA_HPP

#include <glm/glm.hpp>

#include "Device.hpp"

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

    void lookAt(
        const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up);
    void orient(
        const glm::vec3 &eye, const glm::vec3 &fwd, const glm::vec3 &up);
    void perspective(
        const float fov, const float ar, const float zN, const float zF);

    // Move camera orbiting the origin
    void orbit(
        const glm::vec2 &currentPos, const glm::vec2 &lastPos,
        const glm::vec2 &screenCenter);
    void scaleOrbit(
        const float currentY, const float lastY, const float screenCenterY);

    void updateBuffer(const uint32_t index) const;

    std::vector<vk::DescriptorBufferInfo> bufferInfos() const;
    const vk::DescriptorSetLayout &descriptorSetLayout() const;
    const vk::DescriptorSet &descriptorSet(const uint32_t index) const;
    const glm::mat4 &worldToCamera() const;
    const glm::mat4 &cameraToClip() const;

  private:
    void createUniformBuffers(const uint32_t swapImageCount);
    // Create uniform buffers first
    void createDescriptorSets(
        const vk::DescriptorPool descriptorPool, const uint32_t swapImageCount,
        const vk::ShaderStageFlags stageFlags);

    Device *_device = nullptr;
    glm::vec3 _eye;
    glm::mat4 _worldToClip;
    glm::mat4 _worldToCamera;
    glm::mat4 _cameraToClip;

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;
    std::vector<Buffer> _uniformBuffers;
};

#endif // PROSPER_CAMERA_HPP
