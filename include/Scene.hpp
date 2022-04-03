#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include "Camera.hpp"
#include "Light.hpp"
#include "Model.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vector>

struct Scene
{
    struct Node
    {
        std::vector<Node *> children;
        uint32_t modelID{0xFFFFFFFF};
        CameraParameters camera;
        glm::vec3 translation{0.f};
        glm::quat rotation{1.f, 0.f, 0.f, 0.f};
        glm::vec3 scale{1.f};
    };

    struct Lights
    {
        DirectionalLight directionalLight;
        PointLights pointLights;
        SpotLights spotLights;
        std::vector<vk::DescriptorSet> descriptorSets;
        std::vector<vk::DescriptorSet> descriptorSetsClustered;
    };

    CameraParameters camera;

    std::vector<Node *> nodes;

    std::vector<ModelInstance> modelInstances;

    std::vector<Buffer> modelInstanceTransformsBuffers;
    std::vector<vk::DescriptorSet> modelInstancesDescriptorSets;
    vk::DescriptorSet accelerationStructureDS;

    Lights lights;
};

#endif // PROSPER_SCENENODE_HPP
