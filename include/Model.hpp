#ifndef PROSPER_MODEL_HPP
#define PROSPER_MODEL_HPP

#include "Mesh.hpp"

struct Model
{
    struct UBlock {
        glm::mat4 modelToWorld;
    };

    Device* _device;
    std::vector<Mesh> _meshes;
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
        device->logical().mapMemory(uniformBuffers[nextImage].memory, 0, sizeof(UBlock), {}, &data);
        memcpy(data, &uBlock, sizeof(UBlock));
        device->logical().unmapMemory(uniformBuffers[nextImage].memory);
    }
};

#endif // PROSPER_MODEL_HPP
