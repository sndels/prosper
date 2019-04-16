#include "World.hpp"

#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <tiny_gltf.h>

#include <algorithm>
#include <iostream>
#include <set>

#include "Constants.hpp"

using namespace glm;

namespace {
    const std::array<glm::vec3, 36> skyboxVerts{{
        vec3{-1.0f,  1.0f, -1.0f},
        vec3{-1.0f, -1.0f, -1.0f},
        vec3{1.0f, -1.0f, -1.0f},
        vec3{1.0f, -1.0f, -1.0f},
        vec3{1.0f,  1.0f, -1.0f},
        vec3{-1.0f,  1.0f, -1.0f},

        vec3{-1.0f, -1.0f,  1.0f},
        vec3{-1.0f, -1.0f, -1.0f},
        vec3{-1.0f,  1.0f, -1.0f},
        vec3{-1.0f,  1.0f, -1.0f},
        vec3{-1.0f,  1.0f,  1.0f},
        vec3{-1.0f, -1.0f,  1.0f},

        vec3{1.0f, -1.0f, -1.0f},
        vec3{1.0f, -1.0f,  1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{1.0f,  1.0f, -1.0f},
        vec3{1.0f, -1.0f, -1.0f},

        vec3{-1.0f, -1.0f,  1.0f},
        vec3{-1.0f,  1.0f,  1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{1.0f, -1.0f,  1.0f},
        vec3{-1.0f, -1.0f,  1.0f},

        vec3{-1.0f,  1.0f, -1.0f},
        vec3{1.0f,  1.0f, -1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{1.0f,  1.0f,  1.0f},
        vec3{-1.0f,  1.0f,  1.0f},
        vec3{-1.0f,  1.0f, -1.0f},

        vec3{-1.0f, -1.0f, -1.0f},
        vec3{-1.0f, -1.0f,  1.0f},
        vec3{1.0f, -1.0f, -1.0f},
        vec3{1.0f, -1.0f, -1.0f},
        vec3{-1.0f, -1.0f,  1.0f},
        vec3{1.0f, -1.0f,  1.0f}
    }};

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

    Buffer createSkyboxVertexBuffer(Device* device)
    {
        const vk::DeviceSize bufferSize = sizeof(skyboxVerts[0]) * skyboxVerts.size();
        const Buffer stagingBuffer = device->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        void* data;
        device->map(stagingBuffer.allocation, &data);
        memcpy(data, skyboxVerts.data(), static_cast<size_t>(bufferSize));
        device->unmap(stagingBuffer.allocation);

        const auto skyboxVertexBuffer = device->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        device->copyBuffer(stagingBuffer, skyboxVertexBuffer, bufferSize);

        device->destroy(stagingBuffer);
        return skyboxVertexBuffer;
    }
}

World::~World()
{
    _device->logical().destroy(_descriptorPool);
    _device->logical().destroy(_materialDSLayout);
    _device->logical().destroy(_modelInstanceDSLayout);
    _device->destroy(_skyboxVertexBuffer);
    for (auto& scene : _scenes) {
        for (auto& instance: scene.modelInstances) {
            for (auto& buffer : instance.uniformBuffers)
                _device->destroy(buffer);
        }
    }
}

void World::loadGLTF(Device* device, const uint32_t swapImageCount, const std::string& filename)
{
    _device = device;

    {
        auto empty = Texture2D(_device, resPath("texture/empty.png"), false);
        _emptyTexture = std::move(empty);
        auto skybox = TextureCubemap(_device, resPath("env/storm.ktx"));
        _skyboxTexture = std::move(skybox);
        _skyboxVertexBuffer = createSkyboxVertexBuffer(_device);
    }

    const auto gltfModel = loadGLTFModel(filename);

    loadTextures(gltfModel);
    loadMaterials(gltfModel);
    loadModels(gltfModel);
    loadScenes(gltfModel);

    createUniformBuffers(swapImageCount);
    createDescriptorPool(swapImageCount);
    createDescriptorSets(swapImageCount);
}

const Scene& World::currentScene() const
{
    return _scenes[_currentScene];
}

void World::loadTextures(const tinygltf::Model& gltfModel)
{
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
        _textures.emplace_back(_device, image, sampler, true);
    }
}

void World::loadMaterials(const tinygltf::Model& gltfModel)
{
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
            mat._baseColorFactor = make_vec4(elem->second.ColorFactor().data());
        }
        if (const auto& elem = material.values.find("metallicFactor");
            elem != material.values.end()) {
            mat._metallicFactor = elem->second.Factor();
        }
        if (const auto& elem = material.values.find("roughnessFactor");
            elem != material.values.end()) {
            mat._roughnessFactor = elem->second.Factor();
        }
        if (const auto& elem = material.additionalValues.find("alphaMode");
            elem != material.additionalValues.end()) {
            if (elem->second.string_value == "MASK")
                mat._alphaMode = Material::AlphaMode::Mask;
            else if (elem->second.string_value == "BLEND")
                mat._alphaMode = Material::AlphaMode::Blend;
        }
        if (const auto& elem = material.additionalValues.find("alphaCutoff");
            elem != material.additionalValues.end()) {
            mat._alphaCutoff = static_cast<float>(elem->second.Factor());
        }
        // TODO: Support more parameters
        _materials.push_back(std::move(mat));
    }
}

void World::loadModels(const tinygltf::Model& gltfModel)
{
    for (const auto& model : gltfModel.meshes) {
        _models.push_back({_device, {}});
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
            const auto tangents = [&]{
                const auto& attribute = primitive.attributes.find("TANGENT");
                if (attribute == primitive.attributes.end())
                    return static_cast<const float*>(nullptr);
                const auto& accessor = gltfModel.accessors[attribute->second];
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float*>(&(data[offset]));
            }();
            const auto texCoords0 = [&]{
                const auto& attribute = primitive.attributes.find("TEXCOORD_0");
                if (attribute == primitive.attributes.end())
                    return static_cast<const float*>(nullptr);
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
                        vec4{make_vec3(&positions[v * 3]), 1.f},
                        normalize(make_vec3(&normals[v * 3])),
                        tangents ? normalize(make_vec4(&tangents[v * 4])) : vec4(0),
                        texCoords0 ? make_vec2(&texCoords0[v * 2]) : vec2(0)
                    });
                }
                return vs;
            }();

            const std::vector<uint32_t> indices = [&, vertexCount = vertexCount]{
                std::vector<uint32_t> is;
                // TODO: Other index types
                assert(primitive.indices > -1);
                const auto& accessor = gltfModel.accessors[primitive.indices];
                const auto& view = gltfModel.bufferViews[accessor.bufferView];
                const auto& data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;

                if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const auto indexData = reinterpret_cast<const uint32_t*>(&(data[offset]));
                    is = {indexData, indexData + vertexCount};
                } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const auto indexData = reinterpret_cast<const uint16_t*>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                } else {
                    const auto indexData = reinterpret_cast<const uint8_t*>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                }

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
}

void World::loadScenes(const tinygltf::Model& gltfModel)
{
    // TODO: More complex nodes
    _nodes.resize(gltfModel.nodes.size());
    for (size_t n = 0; n < _nodes.size(); ++n) {
        const auto& node = gltfModel.nodes[n];
        std::transform(
            node.children.begin(),
            node.children.end(),
            std::back_inserter(_nodes[n].children),
            [&](int i){ return &_nodes[i]; }
        );
        if (node.mesh > -1)
            _nodes[n].model = &_models[node.mesh];
        if (node.matrix.size() == 16) {
            // Spec defines the matrix to be decomposeable to T * R * S
            const auto matrix = mat4{make_mat4(node.matrix.data())};
            vec3 skew;
            vec4 perspective;
            decompose(
                matrix,
                _nodes[n].scale,
                _nodes[n].rotation,
                _nodes[n].translation,
                skew,
                perspective
            );
        }
        if (node.translation.size() == 3)
            _nodes[n].translation = vec3{make_vec3(node.translation.data())};
        if (node.rotation.size() == 4)
            _nodes[n].rotation = make_quat(node.rotation.data());
        if (node.scale.size() == 3)
            _nodes[n].scale = vec3{make_vec3(node.scale.data())};
    }

    _scenes.resize(gltfModel.scenes.size());
    for (size_t s = 0; s < _scenes.size(); ++s) {
        const auto& scene = gltfModel.scenes[s];
        std::transform(
            scene.nodes.begin(),
            scene.nodes.end(),
            std::back_inserter(_scenes[s].nodes),
            [&](int i){ return &_nodes[i]; }
        );
    }
    _currentScene = max(gltfModel.defaultScene, 0);

    // Traverse scenes and generate model instances for snappier rendering
    std::vector<mat4> parentTransforms{mat4{1.f}};
    for (auto& scene : _scenes) {
        std::set<Scene::Node*> visited;
        std::vector<Scene::Node*> nodeStack = scene.nodes;
        while (!nodeStack.empty()) {
            const auto node = nodeStack.back();
            if (visited.find(node) != visited.end()) {
                nodeStack.pop_back();
                parentTransforms.pop_back();
            } else {
                visited.emplace(node);
                nodeStack.insert(nodeStack.end(), node->children.begin(), node->children.end());
                const mat4 transform =
                    parentTransforms.back() *
                    translate(mat4{1.f}, node->translation) *
                    mat4_cast(node->rotation) *
                    scale(mat4{1.f}, node->scale);
                if (node->model) {
                    scene.modelInstances.push_back({
                        node->model,
                        transform,
                        {},
                        {}
                    });
                }
                parentTransforms.emplace_back(transform);
            }
        }
    }
}

void World::createUniformBuffers(const uint32_t swapImageCount)
{
    const vk::DeviceSize bufferSize = sizeof(Scene::ModelInstance::UBlock);
    for (auto& scene : _scenes) {
        for (auto& modelInstance : scene.modelInstances) {
            for (size_t i = 0; i < swapImageCount; ++i)
                modelInstance.uniformBuffers.push_back(_device->createBuffer(
                    bufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                    VMA_MEMORY_USAGE_CPU_TO_GPU
                ));
        }
    }
}

void World::createDescriptorPool(const uint32_t swapImageCount)
{
    // TODO: Tight bound for node descriptor count by nodes with a mesh
    const std::array<vk::DescriptorPoolSize, 2> poolSizes{{
        { // (Dynamic) Nodes need per frame descriptor sets of one descriptor for the UBlock
            vk::DescriptorType::eUniformBuffer,
            swapImageCount * static_cast<uint32_t>(_nodes.size())// descriptorCount
        },
        { // Materials need one descriptor per texture as they are constant between frames
            vk::DescriptorType::eCombinedImageSampler,
            3 * static_cast<uint32_t>(_materials.size()) // descriptorCount
        }
    }};
    const uint32_t maxSets =
        swapImageCount * static_cast<uint32_t>(_nodes.size()) +
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

    const vk::DescriptorSetLayoutBinding modelInstanceLayoutBinding{
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    };
    _modelInstanceDSLayout = _device->logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &modelInstanceLayoutBinding
    });

    const std::vector<vk::DescriptorSetLayout> modelInstanceLayouts(
        swapImageCount,
        _modelInstanceDSLayout
    );
    for (auto& scene : _scenes) {
        for (auto& instance : scene.modelInstances) {
            instance.descriptorSets = _device->logical().allocateDescriptorSets({
                _descriptorPool,
                static_cast<uint32_t>(modelInstanceLayouts.size()),
                modelInstanceLayouts.data()
            });

            const auto bufferInfos = instance.bufferInfos();
            for (size_t i = 0; i < instance.descriptorSets.size(); ++i) {
                const vk::WriteDescriptorSet descriptorWrite{
                    instance.descriptorSets[i],
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
}
