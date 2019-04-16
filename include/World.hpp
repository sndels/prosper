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
    World() = default;
    ~World();

    World(const World& other) = delete;
    World& operator=(const World& other) = delete;

    void loadGLTF(Device* device, const uint32_t swapImageCount, const std::string& filename);
    const Scene& currentScene() const;

    // TODO: Private?
    std::optional<Texture2D> _emptyTexture;
    std::vector<Texture2D> _textures;
    std::vector<Material> _materials;
    std::vector<Model> _models;
    std::vector<Scene::Node> _nodes;
    std::vector<Scene> _scenes;
    size_t _currentScene;

    std::optional<TextureCubemap> _skyboxTexture;

    vk::DescriptorPool _descriptorPool;
    vk::DescriptorSetLayout _materialDSLayout;
    vk::DescriptorSetLayout _modelInstanceDSLayout;

private:
    void createUniformBuffers(const uint32_t swapImageCount);
    void createDescriptorPool(const uint32_t swapImageCount);
    void createDescriptorSets(const uint32_t swapImageCount);

    Device* _device = nullptr;

};

#endif // PROSPER_WORLD_HPP