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
    static const uint32_t sDirectionalLight = 0xFFFFFFFF;

    // TODO:
    // More cache/memory friendly storage?
    // Shared bitfield for populated members instead of optionals?
    struct Node
    {
        uint32_t firstChild{0};
        uint32_t lastChild{0};
        wheels::Optional<uint32_t> parent;
        wheels::Optional<glm::vec3> translation;
        wheels::Optional<glm::quat> rotation;
        wheels::Optional<glm::vec3> scale;
        wheels::Optional<uint32_t> modelID;
        wheels::Optional<uint32_t> modelInstance;
        wheels::Optional<uint32_t> camera;
        wheels::Optional<uint32_t> pointLight;
        wheels::Optional<uint32_t> spotLight;
        bool directionalLight;
    };

    struct Lights
    {
        DirectionalLight directionalLight;
        PointLights pointLights;
        SpotLights spotLights;
    };

    CameraParameters camera;

    wheels::Array<Node> nodes;
    wheels::Array<uint32_t> rootNodes;
    float endTimeS{0.f};

    wheels::Array<ModelInstance> modelInstances;

    struct RTInstance
    {
        uint32_t modelInstanceID{0};
        uint32_t meshID{0xFFFFFFFF};
        uint32_t materialID{0xFFFFFFFF};
    };
    uint32_t rtInstanceCount{0};
    Buffer rtInstancesBuffer;
    vk::DescriptorSet modelInstancesDescriptorSet;
    vk::DescriptorSet rtDescriptorSet;

    Lights lights;

    Scene(wheels::Allocator &alloc)
    : nodes{alloc}
    , rootNodes{alloc}
    , modelInstances{alloc}
    {
    }
};

#endif // PROSPER_SCENENODE_HPP
