#ifndef PROSPER_SCENE_HPP
#define PROSPER_SCENE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

struct Model;

struct Scene {
  struct Node {
    // TODO: Other fields
    std::vector<Node *> children;
    Model *model = nullptr;
    glm::vec3 translation = glm::vec3{0.f};
    glm::quat rotation = glm::quat{1.f, 0.f, 0.f, 0.f};
    glm::vec3 scale = glm::vec3{1.f};
  };

  struct ModelInstance {
    struct UBlock {
      glm::mat4 modelToWorld;
    };

    Model *model = nullptr;
    glm::mat4 modelToWorld = glm::mat4{1.f};

    std::vector<Buffer> uniformBuffers;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::DescriptorBufferInfo> bufferInfos() const {
      std::vector<vk::DescriptorBufferInfo> infos;
      for (auto &buffer : uniformBuffers)
        infos.emplace_back(buffer.handle, 0, sizeof(UBlock));

      return infos;
    }

    void updateBuffer(const std::shared_ptr<Device> device,
                      const uint32_t nextImage) const {
      UBlock uBlock;
      uBlock.modelToWorld = modelToWorld;

      void *data;
      device->map(uniformBuffers[nextImage].allocation, &data);
      memcpy(data, &uBlock, sizeof(UBlock));
      device->unmap(uniformBuffers[nextImage].allocation);
    }
  };

  std::vector<Node *> nodes;
  std::vector<ModelInstance> modelInstances;
};

#endif // PROSPER_SCENENODE_HPP
