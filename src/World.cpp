#include "World.hpp"

// Define these in exactly one .cpp
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <iostream>

namespace {
    tinygltf::Model loadGLTFModel(const std::string& filename)
    {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string warn;
        std::string err;

        const bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        if (!warn.empty())
            std::cout << "TinyGLTF warning: " << warn << std::endl;
        if (!err.empty())
            std::cerr << "TinyGLTF error: " << err << std::endl;
        if (!ret)
            throw std::runtime_error("Parising glTF failed");

        return model;
    }
}

void World::loadGLTF(Device* device, const std::string& filename)
{
    _device = device;

    const auto gltfModel = loadGLTFModel(filename);

    for (const auto& texture : gltfModel.textures) {
        const auto& image = gltfModel.images[texture.source];
        const tinygltf::Sampler sampler = [&]{
            tinygltf::Sampler s;
            if (texture.sampler == -1) {
                s.minFilter = GL_LINEAR;
                s.magFilter = GL_LINEAR;
                s.wrapS = GL_REPEAT;
                s.wrapT = GL_REPEAT;
            } else
                s = gltfModel.samplers[texture.sampler];
            return s;
        }();
        _textures.emplace_back(_device, image, sampler);
    }
}
