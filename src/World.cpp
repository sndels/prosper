#include "World.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <tiny_gltf.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <algorithm>
#include <iostream>
#include <set>

#include "Utils.hpp"

using namespace glm;

namespace
{
const std::array<glm::vec3, 36> skyboxVerts{
    {vec3{-1.0f, 1.0f, -1.0f},  vec3{-1.0f, -1.0f, -1.0f},
     vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, -1.0f},
     vec3{1.0f, 1.0f, -1.0f},   vec3{-1.0f, 1.0f, -1.0f},

     vec3{-1.0f, -1.0f, 1.0f},  vec3{-1.0f, -1.0f, -1.0f},
     vec3{-1.0f, 1.0f, -1.0f},  vec3{-1.0f, 1.0f, -1.0f},
     vec3{-1.0f, 1.0f, 1.0f},   vec3{-1.0f, -1.0f, 1.0f},

     vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, 1.0f},
     vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
     vec3{1.0f, 1.0f, -1.0f},   vec3{1.0f, -1.0f, -1.0f},

     vec3{-1.0f, -1.0f, 1.0f},  vec3{-1.0f, 1.0f, 1.0f},
     vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
     vec3{1.0f, -1.0f, 1.0f},   vec3{-1.0f, -1.0f, 1.0f},

     vec3{-1.0f, 1.0f, -1.0f},  vec3{1.0f, 1.0f, -1.0f},
     vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
     vec3{-1.0f, 1.0f, 1.0f},   vec3{-1.0f, 1.0f, -1.0f},

     vec3{-1.0f, -1.0f, -1.0f}, vec3{-1.0f, -1.0f, 1.0f},
     vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, -1.0f},
     vec3{-1.0f, -1.0f, 1.0f},  vec3{1.0f, -1.0f, 1.0f}}};

tinygltf::Model loadGLTFModel(const std::string &filename)
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

Buffer createSkyboxVertexBuffer(Device *device)
{
    const vk::DeviceSize bufferSize =
        sizeof(skyboxVerts[0]) * skyboxVerts.size();
    const Buffer stagingBuffer = device->createBuffer(
        "SkyboxVertexStagingBuffer", bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    void *data;
    device->map(stagingBuffer.allocation, &data);
    memcpy(data, skyboxVerts.data(), static_cast<size_t>(bufferSize));
    device->unmap(stagingBuffer.allocation);

    const auto skyboxVertexBuffer = device->createBuffer(
        "SkyboxVertexBuffer", bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY);

    const auto commandBuffer = device->beginGraphicsCommands();

    const vk::BufferCopy copyRegion{
        0, // srcOffset
        0, // dstOffset
        bufferSize};
    commandBuffer.copyBuffer(
        stagingBuffer.handle, skyboxVertexBuffer.handle, 1, &copyRegion);

    device->endGraphicsCommands(commandBuffer);

    device->destroy(stagingBuffer);
    return skyboxVertexBuffer;
}
} // namespace

World::~World()
{
    _device->logical().destroy(_descriptorPool);
    _device->logical().destroy(_dsLayouts.material);
    _device->logical().destroy(_dsLayouts.modelInstance);
    _device->logical().destroy(_dsLayouts.skybox);
    _device->destroy(_skyboxVertexBuffer);
    for (auto &scene : _scenes)
    {
        for (auto &instance : scene.modelInstances)
        {
            for (auto &buffer : instance.uniformBuffers)
                _device->destroy(buffer);
        }
    }
    for (auto &buffer : _skyboxUniformBuffers)
        _device->destroy(buffer);
}

World::World(
    Device *device, const uint32_t swapImageCount, const std::string &filename)
: _emptyTexture{device, resPath("texture/empty.png"), false}
, _skyboxTexture{device, resPath("env/storm.ktx")}
, _skyboxVertexBuffer{createSkyboxVertexBuffer(device)}
, _device{device}
{
    const auto gltfModel = loadGLTFModel(filename);

    loadTextures(gltfModel);
    loadMaterials(gltfModel);
    loadModels(gltfModel);
    loadScenes(gltfModel);

    createUniformBuffers(swapImageCount);
    createDescriptorPool(swapImageCount);
    createDescriptorSets(swapImageCount);
}

const Scene &World::currentScene() const { return _scenes[_currentScene]; }

void World::drawSkybox(const vk::CommandBuffer &buffer) const
{
    const vk::DeviceSize offset = 0;
    buffer.bindVertexBuffers(0, 1, &_skyboxVertexBuffer.handle, &offset);
    buffer.draw(static_cast<uint32_t>(skyboxVerts.size()), 1, 0, 0);
}

void World::loadTextures(const tinygltf::Model &gltfModel)
{
    for (const auto &texture : gltfModel.textures)
    {
        const auto &image = gltfModel.images[texture.source];
        const tinygltf::Sampler sampler = [&]
        {
            tinygltf::Sampler s;
            if (texture.sampler == -1)
            {
                s.minFilter = GL_LINEAR;
                s.magFilter = GL_LINEAR;
                s.wrapS = GL_REPEAT;
                s.wrapT = GL_REPEAT;
            }
            else
                s = gltfModel.samplers[texture.sampler];
            return s;
        }();
        _textures.emplace_back(_device, image, sampler, true);
    }
}

void World::loadMaterials(const tinygltf::Model &gltfModel)
{
    for (const auto &material : gltfModel.materials)
    {
        Material mat;
        if (const auto &elem = material.values.find("baseColorTexture");
            elem != material.values.end())
        {
            mat._baseColor = &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.baseColor = elem->second.TextureTexCoord();
        }
        if (const auto &elem = material.values.find("metallicRoughnessTexture");
            elem != material.values.end())
        {
            mat._metallicRoughness = &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.metallicRoughness =
                elem->second.TextureTexCoord();
        }
        if (const auto &elem = material.additionalValues.find("normalTexture");
            elem != material.additionalValues.end())
        {
            mat._normal = &_textures[elem->second.TextureIndex()];
            mat._texCoordSets.normal = elem->second.TextureTexCoord();
        }
        if (const auto &elem = material.values.find("baseColorFactor");
            elem != material.values.end())
        {
            mat._baseColorFactor = make_vec4(elem->second.ColorFactor().data());
        }
        if (const auto &elem = material.values.find("metallicFactor");
            elem != material.values.end())
        {
            mat._metallicFactor = static_cast<float>(elem->second.Factor());
        }
        if (const auto &elem = material.values.find("roughnessFactor");
            elem != material.values.end())
        {
            mat._roughnessFactor = static_cast<float>(elem->second.Factor());
        }
        if (const auto &elem = material.additionalValues.find("alphaMode");
            elem != material.additionalValues.end())
        {
            if (elem->second.string_value == "MASK")
                mat._alphaMode = Material::AlphaMode::Mask;
            else if (elem->second.string_value == "BLEND")
                mat._alphaMode = Material::AlphaMode::Blend;
        }
        if (const auto &elem = material.additionalValues.find("alphaCutoff");
            elem != material.additionalValues.end())
        {
            mat._alphaCutoff = static_cast<float>(elem->second.Factor());
        }
        // TODO: Support more parameters
        _materials.push_back(std::move(mat));
    }
}

void World::loadModels(const tinygltf::Model &gltfModel)
{
    for (const auto &model : gltfModel.meshes)
    {
        _models.push_back({_device, {}});
        for (const auto &primitive : model.primitives)
        {
            // TODO: More vertex attributes, different modes, no indices
            // Retrieve attribute buffers
            const auto [positions, vertexCount] = [&]
            {
                const auto &attribute = primitive.attributes.find("POSITION");
                assert(attribute != primitive.attributes.end());
                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return std::make_tuple(
                    reinterpret_cast<const float *>(&(data[offset])),
                    accessor.count);
            }();
            const auto normals = [&]
            {
                const auto &attribute = primitive.attributes.find("NORMAL");
                assert(attribute != primitive.attributes.end());
                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float *>(&(data[offset]));
            }();
            const auto tangents = [&]
            {
                const auto &attribute = primitive.attributes.find("TANGENT");
                if (attribute == primitive.attributes.end())
                    return static_cast<const float *>(nullptr);
                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float *>(&(data[offset]));
            }();
            const auto texCoords0 = [&]
            {
                const auto &attribute = primitive.attributes.find("TEXCOORD_0");
                if (attribute == primitive.attributes.end())
                    return static_cast<const float *>(nullptr);
                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float *>(&(data[offset]));
            }();

            // Clang doesn't support capture of structured bindings (yet?)
            const std::vector<Vertex> vertices =
                [&, vertexCount = vertexCount, positions = positions]
            {
                std::vector<Vertex> vs;
                for (size_t v = 0; v < vertexCount; ++v)
                {
                    vs.push_back(
                        {vec4{make_vec3(&positions[v * 3]), 1.f},
                         normalize(make_vec3(&normals[v * 3])),
                         tangents ? normalize(make_vec4(&tangents[v * 4]))
                                  : vec4(0),
                         texCoords0 ? make_vec2(&texCoords0[v * 2]) : vec2(0)});
                }
                return vs;
            }();

            const std::vector<uint32_t> indices = [&, vertexCount = vertexCount]
            {
                std::vector<uint32_t> is;
                // TODO: Other index types
                assert(primitive.indices > -1);
                const auto &accessor = gltfModel.accessors[primitive.indices];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;

                if (accessor.componentType ==
                    TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    const auto indexData =
                        reinterpret_cast<const uint32_t *>(&(data[offset]));
                    is = {indexData, indexData + vertexCount};
                }
                else if (
                    accessor.componentType ==
                    TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    const auto indexData =
                        reinterpret_cast<const uint16_t *>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                }
                else
                {
                    const auto indexData =
                        reinterpret_cast<const uint8_t *>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                }

                return is;
            }();

            const int material = primitive.material;
            assert(material > -1);

            _models.back()._meshes.emplace_back(
                vertices, indices, &_materials[material], _device);
        }
    }
}

void World::loadScenes(const tinygltf::Model &gltfModel)
{
    // TODO: More complex nodes
    _nodes.resize(gltfModel.nodes.size());
    for (size_t n = 0; n < _nodes.size(); ++n)
    {
        const auto &node = gltfModel.nodes[n];
        std::transform(
            node.children.begin(), node.children.end(),
            std::back_inserter(_nodes[n].children),
            [&](int i) { return &_nodes[i]; });
        if (node.mesh > -1)
            _nodes[n].model = &_models[node.mesh];
        if (node.matrix.size() == 16)
        {
            // Spec defines the matrix to be decomposeable to T * R * S
            const auto matrix = mat4{make_mat4(node.matrix.data())};
            vec3 skew;
            vec4 perspective;
            decompose(
                matrix, _nodes[n].scale, _nodes[n].rotation,
                _nodes[n].translation, skew, perspective);
        }
        if (node.translation.size() == 3)
            _nodes[n].translation = vec3{make_vec3(node.translation.data())};
        if (node.rotation.size() == 4)
            _nodes[n].rotation = make_quat(node.rotation.data());
        if (node.scale.size() == 3)
            _nodes[n].scale = vec3{make_vec3(node.scale.data())};
    }

    _scenes.resize(gltfModel.scenes.size());
    for (size_t s = 0; s < _scenes.size(); ++s)
    {
        const auto &scene = gltfModel.scenes[s];
        std::transform(
            scene.nodes.begin(), scene.nodes.end(),
            std::back_inserter(_scenes[s].nodes),
            [&](int i) { return &_nodes[i]; });
    }
    _currentScene = max(gltfModel.defaultScene, 0);

    // Traverse scenes and generate model instances for snappier rendering
    std::vector<mat4> parentTransforms{mat4{1.f}};
    for (auto &scene : _scenes)
    {
        std::set<Scene::Node *> visited;
        std::vector<Scene::Node *> nodeStack = scene.nodes;
        while (!nodeStack.empty())
        {
            const auto node = nodeStack.back();
            if (visited.find(node) != visited.end())
            {
                nodeStack.pop_back();
                parentTransforms.pop_back();
            }
            else
            {
                visited.emplace(node);
                nodeStack.insert(
                    nodeStack.end(), node->children.begin(),
                    node->children.end());
                const mat4 transform = parentTransforms.back() *
                                       translate(mat4{1.f}, node->translation) *
                                       mat4_cast(node->rotation) *
                                       scale(mat4{1.f}, node->scale);
                if (node->model)
                {
                    scene.modelInstances.push_back(
                        {node->model, transform, {}, {}});
                }
                parentTransforms.emplace_back(transform);
            }
        }
    }
}

void World::createUniformBuffers(const uint32_t swapImageCount)
{
    {
        const vk::DeviceSize bufferSize = sizeof(Scene::ModelInstance::UBlock);
        for (auto &scene : _scenes)
        {
            for (auto &modelInstance : scene.modelInstances)
            {
                for (size_t i = 0; i < swapImageCount; ++i)
                    modelInstance.uniformBuffers.push_back(
                        _device->createBuffer(
                            "ModelInstanceUniforms", bufferSize,
                            vk::BufferUsageFlagBits::eUniformBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            VMA_MEMORY_USAGE_CPU_TO_GPU));
            }
        }
    }

    {
        const vk::DeviceSize bufferSize = sizeof(mat4);
        for (size_t i = 0; i < swapImageCount; ++i)
        {
            _skyboxUniformBuffers.push_back(_device->createBuffer(
                "SkyboxUniforms" + std::to_string(i), bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                VMA_MEMORY_USAGE_CPU_TO_GPU));
        }
    }
}

void World::createDescriptorPool(const uint32_t swapImageCount)
{
    // TODO: Tight bound for node descriptor count by nodes with a mesh
    // Skybox cubemap is also one descriptor per image as it's in the same set
    // as camera
    const uint32_t uniformDescriptorCount =
        swapImageCount * (static_cast<uint32_t>(_nodes.size()) + 1);
    const uint32_t samplerDescriptorCount =
        3 * static_cast<uint32_t>(_materials.size()) + swapImageCount;
    const std::array<vk::DescriptorPoolSize, 2> poolSizes{
        {{// (Dynamic) Nodes need per frame descriptor sets of one descriptor
          // for the UBlock
          vk::DescriptorType::eUniformBuffer, uniformDescriptorCount},
         {// Materials need one descriptor per texture as they are constant
          // between frames
          vk::DescriptorType::eCombinedImageSampler, samplerDescriptorCount}}};
    const uint32_t maxSets =
        swapImageCount * (static_cast<uint32_t>(_nodes.size()) + 1) +
        static_cast<uint32_t>(_materials.size()) + swapImageCount;
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()});
}

void World::createDescriptorSets(const uint32_t swapImageCount)
{
    if (_device == nullptr)
        throw std::runtime_error(
            "Tried to create World descriptor sets before loading glTF");

    const std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{
        {{0, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment},
         {1, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment},
         {2, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment}}};
    _dsLayouts.material = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data()});

    for (auto &material : _materials)
    {
        material._descriptorSet = _device->logical().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{
                .descriptorPool = _descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &_dsLayouts.material})[0];

        const std::array<vk::DescriptorImageInfo, 3> imageInfos = [&]
        {
            std::array<vk::DescriptorImageInfo, 3> iis{
                {_emptyTexture.imageInfo(), _emptyTexture.imageInfo(),
                 _emptyTexture.imageInfo()}};
            if (material._baseColor)
                iis[0] = material._baseColor->imageInfo();
            if (material._metallicRoughness)
                iis[1] = material._metallicRoughness->imageInfo();
            if (material._normal)
                iis[2] = material._normal->imageInfo();
            return iis;
        }();

        const std::array<vk::WriteDescriptorSet, 3> writeDescriptorSets = [&]
        {
            std::array<vk::WriteDescriptorSet, 3> dss;
            for (size_t i = 0; i < imageInfos.size(); ++i)
            {
                dss[i].dstSet = material._descriptorSet;
                dss[i].dstBinding = static_cast<uint32_t>(i);
                dss[i].descriptorCount = 1;
                dss[i].descriptorType =
                    vk::DescriptorType::eCombinedImageSampler;
                dss[i].pImageInfo = &imageInfos[i];
            }
            return dss;
        }();

        _device->logical().updateDescriptorSets(
            static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(), 0, nullptr);
    }

    const vk::DescriptorSetLayoutBinding modelInstanceLayoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex};
    _dsLayouts.modelInstance = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1, .pBindings = &modelInstanceLayoutBinding});

    const std::vector<vk::DescriptorSetLayout> modelInstanceLayouts(
        swapImageCount, _dsLayouts.modelInstance);
    for (auto &scene : _scenes)
    {
        for (auto &instance : scene.modelInstances)
        {
            instance.descriptorSets = _device->logical().allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo{
                    .descriptorPool = _descriptorPool,
                    .descriptorSetCount =
                        static_cast<uint32_t>(modelInstanceLayouts.size()),
                    .pSetLayouts = modelInstanceLayouts.data()});

            const auto bufferInfos = instance.bufferInfos();
            for (size_t i = 0; i < instance.descriptorSets.size(); ++i)
            {
                const vk::WriteDescriptorSet descriptorWrite{
                    .dstSet = instance.descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfos[i]};
                _device->logical().updateDescriptorSets(
                    1, &descriptorWrite, 0, nullptr);
            }
        }
    }

    const std::array<vk::DescriptorSetLayoutBinding, 2> skyboxLayoutBindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
    }};
    _dsLayouts.skybox = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(skyboxLayoutBindings.size()),
            .pBindings = skyboxLayoutBindings.data()});

    const std::vector<vk::DescriptorSetLayout> skyboxLayouts(
        swapImageCount, _dsLayouts.skybox);
    _skyboxDSs =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(skyboxLayouts.size()),
            .pSetLayouts = skyboxLayouts.data()});

    const auto skyboxBufferInfos = [&]
    {
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        for (auto &buffer : _skyboxUniformBuffers)
            bufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = buffer.handle, .offset = 0, .range = sizeof(mat4)});
        return bufferInfos;
    }();
    const vk::DescriptorImageInfo skyboxImageInfo = _skyboxTexture.imageInfo();
    for (size_t i = 0; i < _skyboxDSs.size(); ++i)
    {
        const std::array<vk::WriteDescriptorSet, 2> writeDescriptorSets{
            {{.dstSet = _skyboxDSs[i],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &skyboxBufferInfos[i]},
             {.dstSet = _skyboxDSs[i],
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &skyboxImageInfo}}};
        _device->logical().updateDescriptorSets(
            static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(), 0, nullptr);
    }
}
