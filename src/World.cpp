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
    _device->logical().destroy(_modelDSLayout);
    for (auto& model : _models) {
        for (auto& buffer : model.uniformBuffers) {
            _device->logical().destroy(buffer.handle);
            _device->logical().free(buffer.memory);
        }
    }
}

void World::loadGLTF(Device* device, const uint32_t swapImageCount, const std::string& filename)
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

    for (const auto& model : gltfModel.meshes) {
        _models.push_back({_device, {}, {}, {}});
        for (const auto& primitive : model.primitives) {
            // TODO: More vertex attributes, different modes, no indices
            // Retrieve attribute buffers
            const auto [positions, vertexCount] = [&]{
                const auto& attribute = primitive.attributes.find("POSITION");
                assert(attribute != primitive.attributes.end());
                const auto& accessor = gltfModel.accessors[attribute->second];
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return std::make_tuple(
                    reinterpret_cast<const float*>(&(data[offset])),
                    accessor.count
                );
            }();
            const auto normals = [&]{
                const auto& attribute = primitive.attributes.find("NORMAL");
                assert(attribute != primitive.attributes.end());
                const auto& accessor = gltfModel.accessors[attribute->second];
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float*>(&(data[offset]));
            }();
            const auto texCoords0 = [&]{
                const auto& attribute = primitive.attributes.find("TEXCOORD_0");
                assert(attribute != primitive.attributes.end());
                const auto& accessor = gltfModel.accessors[attribute->second];
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float*>(&(data[offset]));
            }();

            // Clang doesn't support capture of structured bindings (yet?)
            const std::vector<Vertex> vertices = [&, vertexCount = vertexCount, positions = positions]{
                std::vector<Vertex> vs;
                for (size_t v = 0; v < vertexCount; ++v) {
                    vs.push_back({
                        glm::vec4{glm::make_vec3(&positions[v * 3]), 1.f},
                        glm::normalize(glm::make_vec3(&normals[v * 3])),
                        glm::make_vec2(&texCoords0[v * 2])
                    });
                }
                return vs;
            }();

            const std::vector<uint32_t> indices = [&]{
                std::vector<uint32_t> is;
                // TODO: Other index types
                assert(primitive.indices > -1);
                const auto& accessor = gltfModel.accessors[primitive.indices];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                const auto* indexData = reinterpret_cast<const uint16_t*>(&(data[offset]));

                is.resize(accessor.count);
                for (size_t i = 0; i < accessor.count; ++i)
                    is[i] = indexData[i];

                return is;
            }();

            const int material = primitive.material;
            assert(material > -1);

            _models.back()._meshes.emplace_back(
                vertices,
                indices,
                &_materials[material],
                _device
            );
        }
    }

    createUniformBuffers(swapImageCount);
    createDescriptorPool(swapImageCount);
    createDescriptorSets(swapImageCount);
}

void World::createUniformBuffers(const uint32_t swapImageCount)
{
    const vk::DeviceSize bufferSize = sizeof(Model::UBlock);
    for (auto& meshInstance : _models) {
        for (size_t i = 0; i < swapImageCount; ++i)
            meshInstance.uniformBuffers.push_back(_device->createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            ));
    }
}

void World::createDescriptorPool(const uint32_t swapImageCount)
{
    const std::array<vk::DescriptorPoolSize, 2> poolSizes{{
        { // (Dynamic) Models need per frame descriptor sets of one descriptor for the UBlock
            vk::DescriptorType::eUniformBuffer,
            swapImageCount * static_cast<uint32_t>(_models.size()) // descriptorCount
        },
        { // Materials need one descriptor per texture as they are constant between frames
            vk::DescriptorType::eCombinedImageSampler,
            3 * static_cast<uint32_t>(_materials.size()) // descriptorCount
        }
    }};
    const uint32_t maxSets =
        swapImageCount * static_cast<uint32_t>(_models.size()) +
        static_cast<uint32_t>(_materials.size());
    _descriptorPool = _device->logical().createDescriptorPool({
        {}, // flags
        maxSets,
        static_cast<uint32_t>(poolSizes.size()),
        poolSizes.data()
    });
}

void World::createDescriptorSets(const uint32_t swapImageCount)
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

    const vk::DescriptorSetLayoutBinding modelLayoutBinding{
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    };
    _modelDSLayout = _device->logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &modelLayoutBinding
    });

    const std::vector<vk::DescriptorSetLayout> meshInstanceLayouts(
        swapImageCount,
        _modelDSLayout
    );
    for (auto& model : _models) {
        model.descriptorSets = _device->logical().allocateDescriptorSets({
            _descriptorPool,
            static_cast<uint32_t>(meshInstanceLayouts.size()),
            meshInstanceLayouts.data()
        });

        const auto bufferInfos = model.bufferInfos();
        for (size_t i = 0; i < model.descriptorSets.size(); ++i) {
            const vk::WriteDescriptorSet descriptorWrite{
                model.descriptorSets[i],
                0, // dstBinding,
                0, // dstArrayElement
                1, // descriptorCount
                vk::DescriptorType::eUniformBuffer,
                nullptr, // pImageInfo
                &bufferInfos[i]
            };
            _device->logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }
    }
}
