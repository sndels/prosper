#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Texture.hpp"

class Device;

class World
{
public:
    World() = default;

    World(const World& other) = delete;
    World operator=(const World& other) = delete;

    void loadGLTF(Device* device, const std::string& filename);

    // TODO: Private?
    std::vector<Texture> _textures;

private:
    Device* _device = nullptr;

};

#endif // PROSPER_WORLD_HPP
