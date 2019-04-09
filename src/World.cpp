#include "World.hpp"

#include <glm/gtc/type_ptr.hpp>

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

    std::string resPath(const std::string& res)
    {
        return std::string{RES_PATH} + res;
    }
}

World::~World()
{
    _device->logical().destroy(_descriptorPool);
    _device->logical().destroy(_materialDSLayout);
}

void World::loadGLTF(Device* device, const std::string& filename)
{
    _device = device;

    {
        auto empty = Texture(_device, resPath("texture/empty.png"));
        _emptyTexture = std::move(empty);
    }

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

    for (const auto& material : gltfModel.materials) {
        Material mat;
        if (const auto& elem = material.values.find("baseColorTexture");
            elem != material.values.end()) {
            mat._baseColor = &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.baseColor = elem->second.TextureTexCoord();
        }
        if (const auto& elem = material.values.find("metallicRoughnessTexture");
            elem != material.values.end()) {
            mat._metallicRoughness= &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.metallicRoughness= elem->second.TextureTexCoord();
        }
        if (const auto& elem = material.additionalValues.find("normalTexture");
            elem != material.additionalValues.end()) {
            mat._normal = &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.normal = elem->second.TextureTexCoord();
        }
        if (const auto& elem = material.values.find("baseColorFactor");
            elem != material.values.end()) {
            mat._baseColorFactor = glm::make_vec4(elem->second.ColorFactor().data());
        }
        if (const auto& elem = material.values.find("metallicFactor");
            elem != material.values.end()) {
            mat._metallicFactor = elem->second.Factor();
        }
        if (const auto& elem = material.values.find("roughnessFactor");
            elem != material.values.end()) {
            mat._roughnessFactor = elem->second.Factor();
        }
        // TODO: Support more parameters
        _materials.push_back(std::move(mat));
    }

    createDescriptorPool();
    createDescriptorSets();
}

void World::createDescriptorPool()
{
    // TODO: See previous implementation in App for multiple pool sizes when needed
    // Materials only need one descriptor per texture as they are constant between frames
    const vk::DescriptorPoolSize poolSize {
        vk::DescriptorType::eCombinedImageSampler,
        3 * static_cast<uint32_t>(_materials.size()) // descriptorCount
    };
    _descriptorPool = _device->logical().createDescriptorPool({
        {}, // flags
        static_cast<uint32_t>(_materials.size()), // maxSets
        1,
        &poolSize
    });
}

void World::createDescriptorSets()
{
    if (_device == nullptr)
        throw std::runtime_error("Tried to create World descriptor sets before loading glTF");

    const std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{{
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
        { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
        { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}
    }};
    _materialDSLayout = _device->logical().createDescriptorSetLayout({
        {}, // flags
        static_cast<uint32_t>(layoutBindings.size()),
        layoutBindings.data()
    });

    for (auto& material : _materials) {
        material._descriptorSet = _device->logical().allocateDescriptorSets({
            _descriptorPool,
            1,
            &_materialDSLayout
        })[0];

        const std::array<vk::DescriptorImageInfo, 3> imageInfos = [&]{
            std::array<vk::DescriptorImageInfo, 3> iis{{
                _emptyTexture.value().imageInfo(),
                _emptyTexture.value().imageInfo(),
                _emptyTexture.value().imageInfo()
            }};
            if (material._baseColor)
                iis[0] = material._baseColor->imageInfo();
            if (material._metallicRoughness)
                iis[1] = material._metallicRoughness->imageInfo();
            if (material._normal)
                iis[2] = material._normal->imageInfo();
            return iis;
        }();

        const std::array<vk::WriteDescriptorSet, 3> writeDescriptorSets = [&]{
            std::array<vk::WriteDescriptorSet, 3> dss;
            for (size_t i = 0; i < imageInfos.size(); ++i) {
                dss[i].dstSet = material._descriptorSet;
                dss[i].dstBinding = static_cast<uint32_t>(i);
                dss[i].descriptorCount = 1;
                dss[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
                dss[i].pImageInfo = &imageInfos[i];
            }
            return dss;
        }();

        _device->logical().updateDescriptorSets(
            static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(),
            0, // descriptorCopyCount
            nullptr  // pDescriptorCopies
        );
    }
}
