#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include "Allocators.hpp"
#include "gfx/Resources.hpp"
#include "scene/Light.hpp"
#include "scene/Model.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>

struct Scene
{
    static const uint32_t sDirectionalLight = 0xFFFF'FFFF;

    // TODO:
    // More cache/memory friendly storage?
    // Shared bitfield for populated members instead of optionals?
    struct Node
    {
        uint32_t gltfSourceNode{0};
        uint32_t firstChild{0};
        uint32_t lastChild{0};
        wheels::Optional<uint32_t> parent;
        wheels::Optional<glm::vec3> translation;
        wheels::Optional<glm::quat> rotation;
        wheels::Optional<glm::vec3> scale;
        wheels::Optional<uint32_t> modelIndex;
        wheels::Optional<uint32_t> modelInstance;
        wheels::Optional<uint32_t> camera;
        wheels::Optional<uint32_t> pointLight;
        wheels::Optional<uint32_t> spotLight;
        bool directionalLight{false};
        // True if either this node's or one of its parents' transform is
        // animated
        bool dynamicTransform{false};
        wheels::StrSpan fullName;
    };

    struct Lights
    {
        DirectionalLight directionalLight;
        PointLights pointLights;
        SpotLights spotLights;
    };

    wheels::Array<Node> nodes{gAllocators.world};
    wheels::Array<wheels::String> fullNodeNames{gAllocators.world};
    wheels::Array<uint32_t> rootNodes{gAllocators.world};
    float endTimeS{0.f};

    wheels::Array<ModelInstance> modelInstances{gAllocators.world};
    bool previousTransformsValid{false};

    uint32_t drawInstanceCount{0};
    Buffer drawInstancesBuffer;
    vk::DescriptorSet sceneInstancesDescriptorSet;
    vk::DescriptorSet rtDescriptorSet;

    Lights lights;
};

#endif // PROSPER_SCENENODE_HPP
