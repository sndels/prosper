#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include "Camera.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <vector>

struct Model;

struct Scene
{
    struct Node
    {
        // TODO: Other fields
        std::vector<Node *> children;
        Model *model = nullptr;
        CameraParameters camera;
        glm::vec3 translation = glm::vec3{0.f};
        glm::quat rotation = glm::quat{1.f, 0.f, 0.f, 0.f};
        glm::vec3 scale = glm::vec3{1.f};
    };

    struct ModelInstance
    {
        struct UBlock
        {
            glm::mat4 modelToWorld;
        };

        Model *model = nullptr;
        glm::mat4 modelToWorld = glm::mat4{1.f};

        std::vector<Buffer> uniformBuffers;
        std::vector<vk::DescriptorSet> descriptorSets;

        std::vector<vk::DescriptorBufferInfo> bufferInfos() const
        {
            std::vector<vk::DescriptorBufferInfo> infos;
            for (auto &buffer : uniformBuffers)
                infos.push_back(vk::DescriptorBufferInfo{
                    .buffer = buffer.handle,
                    .offset = 0,
                    .range = sizeof(UBlock)});

            return infos;
        }

        void updateBuffer(const Device *device, const uint32_t nextImage) const
        {
            UBlock uBlock;
            uBlock.modelToWorld = modelToWorld;

            void *data;
            device->map(uniformBuffers[nextImage].allocation, &data);
            memcpy(data, &uBlock, sizeof(UBlock));
            device->unmap(uniformBuffers[nextImage].allocation);
        }
    };

    struct DirectionalLight
    {
        // Vector types in uniforms need to be aligned to 16 bytes
        struct Parameters
        {
            glm::vec3 irradiance{2.f};
            uint32_t pad;
            glm::vec3 direction{-1.f, -1.f, -1.f};
            uint32_t pad1;
        } parameters;

        std::vector<Buffer> uniformBuffers;
        std::vector<vk::DescriptorSet> descriptorSets;

        std::vector<vk::DescriptorBufferInfo> bufferInfos() const
        {
            std::vector<vk::DescriptorBufferInfo> infos;
            for (auto &buffer : uniformBuffers)
                infos.push_back(vk::DescriptorBufferInfo{
                    .buffer = buffer.handle,
                    .offset = 0,
                    .range = sizeof(Parameters)});

            return infos;
        }

        void updateBuffer(const Device *device, const uint32_t nextImage) const
        {
            void *data;
            device->map(uniformBuffers[nextImage].allocation, &data);
            memcpy(data, &parameters, sizeof(Parameters));
            device->unmap(uniformBuffers[nextImage].allocation);
        }
    };

    CameraParameters camera;
    std::vector<Node *> nodes;
    std::vector<ModelInstance> modelInstances;
    DirectionalLight directionalLight;
};

#endif // PROSPER_SCENENODE_HPP
