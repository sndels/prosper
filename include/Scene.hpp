#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

struct Model;

struct Scene {
    struct Node {
        struct UBlock {
            glm::mat4 modelToWorld;
        };

        // TODO: Other fields
        std::vector<Node*> children;
        Model* model;
        glm::vec3 translation;
        glm::quat rotation;
        glm::vec3 scale;

        Node() :
            children(),
            model(nullptr),
            translation(0.f),
            rotation(1.f, 0.f, 0.f, 0.f),
            scale(1.f)
        { }

        std::vector<Buffer> uniformBuffers;
        std::vector<vk::DescriptorSet> descriptorSets;

        std::vector<vk::DescriptorBufferInfo> bufferInfos() const
        {
            std::vector<vk::DescriptorBufferInfo> infos;
            for (auto& buffer : uniformBuffers)
                infos.emplace_back(buffer.handle, 0, sizeof(UBlock));

            return infos;
        }

        void updateBuffer(const Device* device, const uint32_t nextImage, const glm::mat4& modelToWorld) const
        {
            UBlock uBlock;
            uBlock.modelToWorld = modelToWorld;

            void* data;
            device->map(uniformBuffers[nextImage].allocation, &data);
            memcpy(data, &uBlock, sizeof(UBlock));
            device->unmap(uniformBuffers[nextImage].allocation);
        }
    };

    std::vector<Node*> nodes;
};

#endif // PROSPER_SCENENODE_HPP
