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
    World operator=(const World& other) = delete;

    void loadGLTF(Device* device, const uint32_t swapImageCount, const std::string& filename);
    void createUniformBuffers(const uint32_t swapImageCount);
    void createDescriptorPool(const uint32_t swapImageCount);
    void createDescriptorSets(const uint32_t swapImageCount);

    // TODO: Private?
    std::optional<Texture> _emptyTexture;
    std::vector<Texture> _textures;
    std::vector<Material> _materials;
    std::vector<Model> _models;
    std::vector<Scene::Node> _nodes;
    std::vector<Scene> _scenes;

    vk::DescriptorPool _descriptorPool;
    vk::DescriptorSetLayout _materialDSLayout;
    vk::DescriptorSetLayout _nodeDSLayout;

private:
    Device* _device = nullptr;

};

#endif // PROSPER_WORLD_HPP
