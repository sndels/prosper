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
        struct Parameters
        {
            // Use vec4 because vec3 alignment is no fun between glsl, c++
            glm::vec4 irradiance{2.f};
            glm::vec4 direction{-1.f, -1.f, -1.f, 1.f};
        } parameters;

        std::vector<Buffer> uniformBuffers;

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

    struct PointLights
    {
        static const uint32_t max_count = 1000000;
        struct PointLight
        {
            glm::vec4 radiance;
            glm::vec4 position;
        };

        struct BufferData
        {
            PointLight lights[max_count];
            uint32_t count{0};
        } bufferData;

        std::vector<Buffer> storageBuffers;

        std::vector<vk::DescriptorBufferInfo> bufferInfos() const
        {
            std::vector<vk::DescriptorBufferInfo> infos;
            for (auto &buffer : storageBuffers)
                infos.push_back(vk::DescriptorBufferInfo{
                    .buffer = buffer.handle,
                    .offset = 0,
                    .range = sizeof(PointLights::BufferData)});

            return infos;
        }

        void updateBuffer(const Device *device, const uint32_t nextImage) const
        {
            void *data;
            device->map(storageBuffers[nextImage].allocation, &data);
            memcpy(data, &bufferData, sizeof(PointLights::BufferData));
            device->unmap(storageBuffers[nextImage].allocation);
        }
    };

    struct Lights
    {
        DirectionalLight directionalLight;
        PointLights pointLights;
        std::vector<vk::DescriptorSet> descriptorSets;
    };

    CameraParameters camera;
    std::vector<Node *> nodes;
    std::vector<ModelInstance> modelInstances;
    Lights lights;
};

#endif // PROSPER_SCENENODE_HPP
