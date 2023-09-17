#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include "Camera.hpp"
#include "Light.hpp"
#include "Model.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

struct Scene
{
    struct Node
    {
        wheels::Array<Node *> children;
        uint32_t modelID{0xFFFFFFFF};
        CameraParameters camera;
        glm::vec3 translation{0.f};
        glm::quat rotation{1.f, 0.f, 0.f, 0.f};
        glm::vec3 scale{1.f};

        Node(wheels::Allocator &alloc)
        : children{alloc}
        {
        }
    };

    struct Lights
    {
        DirectionalLight directionalLight;
        PointLights pointLights;
        SpotLights spotLights;
        wheels::Array<vk::DescriptorSet> descriptorSets;
        wheels::Array<vk::DescriptorSet> descriptorSetsClustered;

        Lights(wheels::Allocator &alloc)
        : descriptorSets{alloc}
        , descriptorSetsClustered{alloc}
        {
        }
    };

    CameraParameters camera;

    wheels::Array<Node *> nodes;

    wheels::Array<ModelInstance> modelInstances;

    struct RTInstance
    {
        uint32_t modelInstanceID{0};
        uint32_t meshID{0xFFFFFFFF};
        uint32_t materialID{0xFFFFFFFF};
    };
    uint32_t rtInstanceCount{0};
    Buffer rtInstancesBuffer;
    wheels::Array<Buffer> modelInstanceTransformsBuffers;
    wheels::Array<vk::DescriptorSet> modelInstancesDescriptorSets;
    vk::DescriptorSet rtDescriptorSet;

    Lights lights;

    Scene(wheels::Allocator &alloc)
    : nodes{alloc}
    , modelInstances{alloc}
    , modelInstanceTransformsBuffers{alloc}
    , modelInstancesDescriptorSets{alloc}
    , lights{alloc}
    {
    }
};

#endif // PROSPER_SCENENODE_HPP