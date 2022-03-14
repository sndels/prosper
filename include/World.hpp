#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "Texture.hpp"

#include <filesystem>
#include <unordered_map>

namespace tinygltf
{
class Model;
};

class Device;

class World
{
  public:
    struct DSLayouts
    {
        vk::DescriptorSetLayout materialTextures;
        vk::DescriptorSetLayout modelInstances;
        vk::DescriptorSetLayout lights;
        vk::DescriptorSetLayout lightsClustered;
        vk::DescriptorSetLayout skybox;
    };
    World(
        Device *device, uint32_t swapImageCount,
        const std::filesystem::path &scene);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    [[nodiscard]] const Scene &currentScene() const;
    void updateUniformBuffers(const Camera &cam, uint32_t nextImage) const;
    void drawSkybox(const vk::CommandBuffer &buffer) const;

    // TODO: Private?
    Texture2D _emptyTexture;
    TextureCubemap _skyboxTexture;
    Buffer _skyboxVertexBuffer;

    std::unordered_map<Scene::Node *, CameraParameters> _cameras;
    std::vector<Texture2D> _textures;
    std::vector<Material> _materials;
    std::vector<Mesh> _meshes;
    std::vector<Model> _models;
    std::vector<Scene::Node> _nodes;
    std::vector<Scene> _scenes;
    size_t _currentScene{0};

    vk::DescriptorPool _descriptorPool;
    Buffer _materialsBuffer;
    vk::DescriptorSet _materialTexturesDS;
    DSLayouts _dsLayouts;

    std::vector<Buffer> _skyboxUniformBuffers;
    std::vector<vk::DescriptorSet> _skyboxDSs;

  private:
    void loadTextures(const tinygltf::Model &gltfModel);
    void loadMaterials(const tinygltf::Model &gltfModel);
    void loadModels(const tinygltf::Model &gltfModel);
    void loadScenes(const tinygltf::Model &gltfModel);
    void createBuffers(uint32_t swapImageCount);
    void createDescriptorPool(uint32_t swapImageCount);
    void createDescriptorSets(uint32_t swapImageCount);

    Device *_device{nullptr};
};

#endif // PROSPER_WORLD_HPP
