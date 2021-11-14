#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "Texture.hpp"

class Device;

class World
{
  public:
    struct DSLayouts
    {
        vk::DescriptorSetLayout material;
        vk::DescriptorSetLayout modelInstance;
        vk::DescriptorSetLayout skybox;
    };
    World(
        std::shared_ptr<Device> device, const uint32_t swapImageCount,
        const std::string &filename);
    ~World();

    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    const Scene &currentScene() const;
    void drawSkybox(const vk::CommandBuffer &buffer) const;

    // TODO: Private?
    Texture2D _emptyTexture;
    TextureCubemap _skyboxTexture;
    Buffer _skyboxVertexBuffer;

    std::vector<Texture2D> _textures;
    std::vector<Material> _materials;
    std::vector<Model> _models;
    std::vector<Scene::Node> _nodes;
    std::vector<Scene> _scenes;
    size_t _currentScene = 0;

    vk::DescriptorPool _descriptorPool;
    DSLayouts _dsLayouts;

    std::vector<Buffer> _skyboxUniformBuffers;
    std::vector<vk::DescriptorSet> _skyboxDSs;

  private:
    void loadTextures(const tinygltf::Model &gltfModel);
    void loadMaterials(const tinygltf::Model &gltfModel);
    void loadModels(const tinygltf::Model &gltfModel);
    void loadScenes(const tinygltf::Model &gltfModel);
    void createUniformBuffers(const uint32_t swapImageCount);
    void createDescriptorPool(const uint32_t swapImageCount);
    void createDescriptorSets(const uint32_t swapImageCount);

    std::shared_ptr<Device> _device = nullptr;
};

#endif // PROSPER_WORLD_HPP
