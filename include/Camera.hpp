#ifndef PROSPER_CAMERA_HPP
#define PROSPER_CAMERA_HPP

#include <glm/glm.hpp>

#include "Device.hpp"

struct CameraUniforms {
    glm::mat4 worldToCamera;
    glm::mat4 cameraToClip;
};

class Camera {
public:
    Camera() = default;
    Camera(Device* device, const uint32_t bufferCount, const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up, const float fov, const float ar, const float zN, const float zF);
    ~Camera();

    void createUniformBuffers(Device* device, const uint32_t count);

    void lookAt(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up);
    void orient(const glm::vec3& eye, const glm::vec3& fwd, const glm::vec3& up);
    void perspective(const float fov, const float ar, const float zN, const float zF);
    void updateBuffer(const uint32_t index) const;

    std::vector<vk::DescriptorBufferInfo> bufferInfos() const;

private:
    void destroyUniformBuffers();

    Device* _device = nullptr;
    std::vector<Buffer> _uniformBuffers;
    glm::mat4 _worldToClip;
    glm::mat4 _worldToCamera;
    glm::mat4 _cameraToClip;

};

#endif // PROSPER_CAMERA_HPP
