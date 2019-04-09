#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Texture.hpp"

class Device;

class World
{
public:
    World() = default;
    ~World();

    World(const World& other) = delete;
    World operator=(const World& other) = delete;

    void loadGLTF(Device* device, const std::string& filename);
    void createDescriptorPool();
    void createDescriptorSets();

    // TODO: Private?
    std::optional<Texture> _emptyTexture;
    std::vector<Texture> _textures;
    std::vector<Material> _materials;

    vk::DescriptorPool _descriptorPool;
    vk::DescriptorSetLayout _materialDSLayout;

private:
    Device* _device = nullptr;

};

#endif // PROSPER_WORLD_HPP
