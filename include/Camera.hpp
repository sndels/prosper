#ifndef PROSPER_CAMERA_HPP
#define PROSPER_CAMERA_HPP

#include <glm/glm.hpp>

class Camera {
public:
    Camera() {};
    Camera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up, float fov, float ar, float zN, float zF);
    ~Camera() {};

    void lookAt(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up);
    void orient(const glm::vec3& eye, const glm::vec3& fwd, const glm::vec3& up);
    void perspective(float fov, float ar, float zN, float zF);

    const glm::mat4& worldToClip() const;

private:
    glm::mat4 _worldToClip;
    glm::mat4 _worldToCamera;
    glm::mat4 _cameraToClip;

};

#endif // PROSPER_CAMERA_HPP
