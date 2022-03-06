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

#include "Timer.hpp"
#include "Utils.hpp"

using namespace glm;

namespace
{

const size_t SKYBOX_VERTS_SIZE = 36;

tinygltf::Model loadGLTFModel(const std::filesystem::path &path)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn;
    std::string err;

    bool ret = false;
    if (path.extension() == ".gltf")
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    else if (path.extension() == ".glb")
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    else
        throw std::runtime_error(
            "Unknown extension '" + path.extension().string() + "'");
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
    // Avoid large global allocation
    const std::array<glm::vec3, SKYBOX_VERTS_SIZE> skyboxVerts{
        vec3{-1.0f, 1.0f, -1.0f},  vec3{-1.0f, -1.0f, -1.0f},
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
        vec3{-1.0f, -1.0f, 1.0f},  vec3{1.0f, -1.0f, 1.0f},
    };

    const vk::DeviceSize bufferSize =
        sizeof(skyboxVerts[0]) * skyboxVerts.size();
    const Buffer stagingBuffer = device->createBuffer(
        "SkyboxVertexStagingBuffer", bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    void *data = nullptr;
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
        bufferSize,
    };
    commandBuffer.copyBuffer(
        stagingBuffer.handle, skyboxVertexBuffer.handle, 1, &copyRegion);

    device->endGraphicsCommands(commandBuffer);

    device->destroy(stagingBuffer);
    return skyboxVertexBuffer;
}
} // namespace

World::World(
    Device *device, const uint32_t swapImageCount,
    const std::filesystem::path &scene)
: _emptyTexture{device, resPath("texture/empty.png"), false}
, _skyboxTexture{device, resPath("env/storm.ktx")}
, _skyboxVertexBuffer{createSkyboxVertexBuffer(device)}
, _device{device}
{
    fprintf(stderr, "Loading world\n");

    Timer t;
    const auto gltfModel = loadGLTFModel(resPath(scene));
    fprintf(stderr, "glTF model loading took %.2fs\n", t.getSeconds());

    const auto &tl = [&](const char *stage, std::function<void()> const &fn)
    {
        t.reset();
        fn();
        fprintf(stderr, "%s took %.2fs\n", stage, t.getSeconds());
    };

    tl("Texture loading", [&]() { loadTextures(gltfModel); });
    tl("Material loading", [&]() { loadMaterials(gltfModel); });
    tl("Model loading ", [&]() { loadModels(gltfModel); });
    tl("Scene loading ", [&]() { loadScenes(gltfModel); });

    tl("Buffer creation", [&]() { createBuffers(swapImageCount); });

    createDescriptorPool(swapImageCount);
    createDescriptorSets(swapImageCount);
}

World::~World()
{
    _device->logical().destroy(_descriptorPool);
    _device->logical().destroy(_dsLayouts.materialTextures);
    _device->logical().destroy(_dsLayouts.modelInstances);
    _device->logical().destroy(_dsLayouts.lights);
    _device->logical().destroy(_dsLayouts.skybox);
    _device->destroy(_skyboxVertexBuffer);
    _device->destroy(_materialsBuffer);
    for (auto &scene : _scenes)
    {
        for (auto &buffer : scene.modelInstanceTransformsBuffers)
            _device->destroy(buffer);

        for (auto &buffer : scene.lights.directionalLight.uniformBuffers)
            _device->destroy(buffer);

        for (auto &buffer : scene.lights.pointLights.storageBuffers)
            _device->destroy(buffer);

        for (auto &buffer : scene.lights.spotLights.storageBuffers)
            _device->destroy(buffer);
    }
    for (auto &buffer : _skyboxUniformBuffers)
        _device->destroy(buffer);
}

const Scene &World::currentScene() const { return _scenes[_currentScene]; }

void World::updateUniformBuffers(
    const Camera &cam, const uint32_t nextImage) const
{
    {
        const mat4 worldToClip =
            cam.cameraToClip() * mat4(mat3(cam.worldToCamera()));
        void *data = nullptr;
        _device->map(_skyboxUniformBuffers[nextImage].allocation, &data);
        memcpy(data, &worldToClip, sizeof(mat4));
        _device->unmap(_skyboxUniformBuffers[nextImage].allocation);
    }

    const auto &scene = currentScene();

    {
        std::vector<Scene::ModelInstance::Transforms> transforms;
        transforms.reserve(scene.modelInstances.size());
        for (const auto &instance : scene.modelInstances)
            transforms.push_back(instance.transforms);

        auto *allocation =
            scene.modelInstanceTransformsBuffers[nextImage].allocation;
        void *data = nullptr;
        _device->map(allocation, &data);
        memcpy(
            data, transforms.data(),
            sizeof(Scene::ModelInstance::Transforms) * transforms.size());
        _device->unmap(allocation);
    }

    scene.lights.directionalLight.updateBuffer(_device, nextImage);
    scene.lights.pointLights.updateBuffer(_device, nextImage);
    scene.lights.spotLights.updateBuffer(_device, nextImage);
}

void World::drawSkybox(const vk::CommandBuffer &buffer) const
{
    const vk::DeviceSize offset = 0;
    buffer.bindVertexBuffers(0, 1, &_skyboxVertexBuffer.handle, &offset);
    buffer.draw(static_cast<uint32_t>(SKYBOX_VERTS_SIZE), 1, 0, 0);
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
    _materials.push_back(Material{});

    for (const auto &material : gltfModel.materials)
    {
        Material mat;
        if (const auto &elem = material.values.find("baseColorTexture");
            elem != material.values.end())
        {
            mat.baseColor = elem->second.TextureIndex() + 1;
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Base color TexCoord isn't 0\n",
                    material.name.c_str());
        }
        if (const auto &elem = material.values.find("metallicRoughnessTexture");
            elem != material.values.end())
        {
            mat.metallicRoughness = elem->second.TextureIndex() + 1;
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Metallic roughness TexCoord isn't 0\n",
                    material.name.c_str());
        }
        if (const auto &elem = material.additionalValues.find("normalTexture");
            elem != material.additionalValues.end())
        {
            mat.normal = elem->second.TextureIndex() + 1;
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Normal TexCoord isn't 0\n",
                    material.name.c_str());
        }
        if (const auto &elem = material.values.find("baseColorFactor");
            elem != material.values.end())
        {
            mat.baseColorFactor = make_vec4(elem->second.ColorFactor().data());
        }
        if (const auto &elem = material.values.find("metallicFactor");
            elem != material.values.end())
        {
            mat.metallicFactor = static_cast<float>(elem->second.Factor());
        }
        if (const auto &elem = material.values.find("roughnessFactor");
            elem != material.values.end())
        {
            mat.roughnessFactor = static_cast<float>(elem->second.Factor());
        }
        if (const auto &elem = material.additionalValues.find("alphaMode");
            elem != material.additionalValues.end())
        {
            if (elem->second.string_value == "MASK")
                mat.alphaMode = Material::AlphaMode::Mask;
            else if (elem->second.string_value == "BLEND")
                mat.alphaMode = Material::AlphaMode::Blend;
        }
        if (const auto &elem = material.additionalValues.find("alphaCutoff");
            elem != material.additionalValues.end())
        {
            mat.alphaCutoff = static_cast<float>(elem->second.Factor());
        }
        _materials.push_back(mat);
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
            const auto *normals = [&]
            {
                const auto &attribute = primitive.attributes.find("NORMAL");
                assert(attribute != primitive.attributes.end());
                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto &data = gltfModel.buffers[view.buffer].data;
                const size_t offset = accessor.byteOffset + view.byteOffset;
                return reinterpret_cast<const float *>(&(data[offset]));
            }();
            const auto *tangents = [&]
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
            const auto *texCoords0 = [&]
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
                    vs.push_back(Vertex{
                        .pos = vec4{make_vec3(&positions[v * 3]), 1.f},
                        .normal = normalize(make_vec3(&normals[v * 3])),
                        .tangent = tangents != nullptr
                                       ? normalize(make_vec4(&tangents[v * 4]))
                                       : vec4(0),
                        .texCoord0 = texCoords0 != nullptr
                                         ? make_vec2(&texCoords0[v * 2])
                                         : vec2(0),
                    });
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
                    const auto *indexData =
                        reinterpret_cast<const uint32_t *>(&(data[offset]));
                    is = {indexData, indexData + vertexCount};
                }
                else if (
                    accessor.componentType ==
                    TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    const auto *indexData =
                        reinterpret_cast<const uint16_t *>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                }
                else
                {
                    const auto *indexData =
                        reinterpret_cast<const uint8_t *>(&(data[offset]));
                    is.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                        is[i] = indexData[i];
                }

                return is;
            }();

            // -1 is mapped to the default material
            assert(primitive.material > -2);
            const uint32_t material = primitive.material + 1;

            _models.back()._meshes.emplace_back(
                vertices, indices, material, _device);
        }
    }
}

void World::loadScenes(const tinygltf::Model &gltfModel)
{
    std::unordered_map<Scene::Node *, size_t> lights;
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
        if (node.camera > -1)
        {
            const auto &cam = gltfModel.cameras[node.camera];
            if (cam.type == "perspective")
                _cameras[&_nodes[n]] = CameraParameters{
                    .fov = static_cast<float>(cam.perspective.yfov),
                    .zN = static_cast<float>(cam.perspective.znear),
                    .zF = static_cast<float>(cam.perspective.zfar),
                };
            else
                fprintf(
                    stderr, "Camera type '%s' is not supported\n",
                    cam.type.c_str());
        }
        if (node.extensions.contains("KHR_lights_punctual"))
        {
            // operator[] doesn't work for some reason
            const auto &ext = node.extensions.at("KHR_lights_punctual");
            const auto &obj = ext.Get<tinygltf::Value::Object>();

            const auto &light = obj.find("light")->second;
            assert(light.IsInt());

            lights[&_nodes[n]] = static_cast<size_t>(light.GetNumberAsInt());
        }
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
        bool directionalLightFound = false;
        std::set<Scene::Node *> visited;
        std::vector<Scene::Node *> nodeStack = scene.nodes;
        while (!nodeStack.empty())
        {
            auto *node = nodeStack.back();
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
                const mat4 modelToWorld =
                    parentTransforms.back() *
                    translate(mat4{1.f}, node->translation) *
                    mat4_cast(node->rotation) * scale(mat4{1.f}, node->scale);
                if (node->model != nullptr)
                {
                    scene.modelInstances.push_back(
                        {.id =
                             static_cast<uint32_t>(scene.modelInstances.size()),
                         .model = node->model,
                         .transforms = {
                             .modelToWorld = modelToWorld,
                         }});
                }
                if (_cameras.contains(node))
                {
                    scene.camera = _cameras[node];
                    scene.camera.eye =
                        vec3{modelToWorld * vec4{0.f, 0.f, 0.f, 1.f}};
                    // TODO: Halfway from camera to scene bb end if inside bb /
                    // halfway of bb if outside of bb?
                    scene.camera.target =
                        vec3{modelToWorld * vec4{0.f, 0.f, -1.f, 1.f}};
                    scene.camera.up = mat3{modelToWorld} * vec3{0.f, 1.f, 0.f};
                }
                if (lights.contains(node))
                {
                    const auto &light = gltfModel.lights[lights[node]];
                    if (light.type == "directional")
                    {
                        if (directionalLightFound)
                        {
                            fprintf(
                                stderr,
                                "Found second directional light for a scene."
                                " Ignoring since only one is supported\n");
                        }
                        auto &parameters =
                            scene.lights.directionalLight.parameters;
                        // gltf blender exporter puts W/m^2 into intensity
                        parameters.irradiance =
                            vec4{
                                static_cast<float>(light.color[0]),
                                static_cast<float>(light.color[1]),
                                static_cast<float>(light.color[2]), 0.f} *
                            static_cast<float>(light.intensity);
                        parameters.direction = vec4{
                            mat3{modelToWorld} * vec3{0.f, 0.f, -1.f}, 0.f};
                        directionalLightFound = true;
                    }
                    else if (light.type == "point")
                    {
                        auto radiance =
                            vec3{
                                static_cast<float>(light.color[0]),
                                static_cast<float>(light.color[1]),
                                static_cast<float>(light.color[2])} *
                            static_cast<float>(light.intensity)
                            // gltf blender exporter puts W into intensity
                            / (4.f * glm::pi<float>());
                        const auto luminance =
                            dot(radiance, vec3{0.2126, 0.7152, 0.0722});
                        const auto minLuminance = 0.01f;
                        const auto radius =
                            light.range > 0.f ? light.range
                                              : sqrt(luminance / minLuminance);

                        auto &data = scene.lights.pointLights.bufferData;
                        const auto i = data.count++;
                        auto &sceneLight = data.lights[i];
                        sceneLight.radianceAndRadius = vec4{radiance, radius};
                        sceneLight.position =
                            modelToWorld * vec4{0.f, 0.f, 0.f, 1.f};
                    }
                    else if (light.type == "spot")
                    {
                        auto &data = scene.lights.spotLights.bufferData;
                        const auto i = data.count++;

                        // Angular attenuation rom gltf spec
                        const auto angleScale =
                            1.f /
                            max(0.001f, static_cast<float>(
                                            cos(light.spot.innerConeAngle) -
                                            cos(light.spot.outerConeAngle)));
                        const auto angleOffset =
                            static_cast<float>(
                                -cos(light.spot.outerConeAngle)) *
                            angleScale;

                        auto &sceneLight = data.lights[i];
                        sceneLight.radianceAndAngleScale =
                            vec4{
                                static_cast<float>(light.color[0]),
                                static_cast<float>(light.color[1]),
                                static_cast<float>(light.color[2]), 0.f} *
                            static_cast<float>(light.intensity);
                        // gltf blender exporter puts W into intensity
                        sceneLight.radianceAndAngleScale /=
                            4.f * glm::pi<float>();
                        sceneLight.radianceAndAngleScale.w = angleScale;

                        sceneLight.positionAndAngleOffset =
                            modelToWorld * vec4{0.f, 0.f, 0.f, 1.f};
                        sceneLight.positionAndAngleOffset.w = angleOffset;

                        sceneLight.direction = vec4{
                            mat3{modelToWorld} * vec3{0.f, 0.f, -1.f}, 0.f};
                    }
                    else
                    {
                        fprintf(
                            stderr, "Unknown light type '%s'\n",
                            light.type.c_str());
                    }
                }
                parentTransforms.emplace_back(modelToWorld);
            }
        }

        // Honor scene lighting
        if (!directionalLightFound &&
            (scene.lights.pointLights.bufferData.count > 0 ||
             scene.lights.spotLights.bufferData.count > 0))
        {
            scene.lights.directionalLight.parameters.irradiance = vec4{0.f};
        }
    }
}

void World::createBuffers(const uint32_t swapImageCount)
{
    {
        const vk::DeviceSize bufferSize =
            _materials.size() * sizeof(_materials[0]);
        const Buffer stagingBuffer = _device->createBuffer(
            "MaterialsStagingBuffer", bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            VMA_MEMORY_USAGE_CPU_TO_GPU);

        void *data = nullptr;
        _device->map(stagingBuffer.allocation, &data);
        memcpy(data, _materials.data(), static_cast<size_t>(bufferSize));
        _device->unmap(stagingBuffer.allocation);

        _materialsBuffer = _device->createBuffer(
            "MaterialsBuffer", bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY);

        const auto commandBuffer = _device->beginGraphicsCommands();

        const vk::BufferCopy copyRegion{
            0, // srcOffset
            0, // dstOffset
            bufferSize};
        commandBuffer.copyBuffer(
            stagingBuffer.handle, _materialsBuffer.handle, 1, &copyRegion);

        _device->endGraphicsCommands(commandBuffer);

        _device->destroy(stagingBuffer);
    }

    {
        for (auto &scene : _scenes)
        {
            {
                const vk::DeviceSize bufferSize =
                    sizeof(Scene::ModelInstance::Transforms) *
                    scene.modelInstances.size();
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.modelInstanceTransformsBuffers.push_back(
                        _device->createBuffer(
                            "instanceTransforms", bufferSize,
                            vk::BufferUsageFlagBits::eStorageBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            VMA_MEMORY_USAGE_CPU_TO_GPU));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(Scene::DirectionalLight::Parameters);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.directionalLight.uniformBuffers.push_back(
                        _device->createBuffer(
                            "DirectionalLightUniforms", bufferSize,
                            vk::BufferUsageFlagBits::eUniformBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            VMA_MEMORY_USAGE_CPU_TO_GPU));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(Scene::PointLights::BufferData);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.pointLights.storageBuffers.push_back(
                        _device->createBuffer(
                            "PointLightsBuffer", bufferSize,
                            vk::BufferUsageFlagBits::eStorageBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            VMA_MEMORY_USAGE_CPU_TO_GPU));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(Scene::SpotLights::BufferData);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.spotLights.storageBuffers.push_back(
                        _device->createBuffer(
                            "SpotLightsBuffer", bufferSize,
                            vk::BufferUsageFlagBits::eStorageBuffer,
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
    // Skybox cubemap is also one descriptor per image
    // As is the directional light
    const uint32_t uniformDescriptorCount =
        swapImageCount * (static_cast<uint32_t>(_nodes.size()) + 2);
    const uint32_t samplerDescriptorCount =
        3 * static_cast<uint32_t>(_materials.size()) + swapImageCount;
    const std::array<vk::DescriptorPoolSize, 3> poolSizes{
        {{// Dynamic need per frame descriptor sets of one descriptor per UBlock
          vk::DescriptorType::eUniformBuffer, uniformDescriptorCount},
         {// Materials need one descriptor per texture as they are constant
          // between frames
          vk::DescriptorType::eCombinedImageSampler, samplerDescriptorCount},
         {// Lights require per frame descriptors for points, spots
          vk::DescriptorType::eStorageBuffer, 2 * swapImageCount}},
    };
    // Per-frame: Nodes, skybox, dirlight, points and spots
    // Single: Materials
    const uint32_t maxSets =
        swapImageCount * ((static_cast<uint32_t>(_nodes.size()) + 3)) +
        static_cast<uint32_t>(_materials.size());
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });
}

void World::createDescriptorSets(const uint32_t swapImageCount)
{
    if (_device == nullptr)
        throw std::runtime_error(
            "Tried to create World descriptor sets before loading glTF");

    {
        std::vector<vk::DescriptorImageInfo> infos;
        infos.reserve(_textures.size() + 1);
        infos.push_back(_emptyTexture.imageInfo());
        for (const auto &tex : _textures)
            infos.push_back(tex.imageInfo());
        const auto infoCount = static_cast<uint32_t>(infos.size());

        std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = infoCount,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        std::array<vk::DescriptorBindingFlags, 2> layoutFlags{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlagBits::eVariableDescriptorCount,
        };
        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            .bindingCount = static_cast<uint32_t>(layoutFlags.size()),
            .pBindingFlags = layoutFlags.data(),
        };
        _dsLayouts.materialTextures =
            _device->logical().createDescriptorSetLayout(
                vk::DescriptorSetLayoutCreateInfo{
                    .pNext = &flagsInfo,
                    .bindingCount =
                        static_cast<uint32_t>(layoutBindings.size()),
                    .pBindings = layoutBindings.data(),
                });

        vk::DescriptorSetVariableDescriptorCountAllocateInfo vainfo{
            .descriptorSetCount = 1,
            .pDescriptorCounts = &infoCount,
        };
        _materialTexturesDS = _device->logical().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{
                .pNext = &vainfo,
                .descriptorPool = _descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &_dsLayouts.materialTextures})[0];

        vk::DescriptorBufferInfo datasInfo{
            .buffer = _materialsBuffer.handle,
            .range = VK_WHOLE_SIZE,
        };

        std::vector<vk::WriteDescriptorSet> dss;
        dss.reserve(infos.size() + 1);
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _materialTexturesDS,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &datasInfo,
        });
        for (uint32_t i = 0; i < infos.size(); ++i)
            dss.push_back(vk::WriteDescriptorSet{
                .dstSet = _materialTexturesDS,
                .dstBinding = 1,
                .dstArrayElement = i,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &infos[i],
            });

        _device->logical().updateDescriptorSets(
            static_cast<uint32_t>(dss.size()), dss.data(), 0, nullptr);
    }
    {
        const vk::DescriptorSetLayoutBinding layoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };
        vk::DescriptorBindingFlags layoutFlags{};
        vk::StructureChain<
            vk::DescriptorSetLayoutCreateInfo,
            vk::DescriptorSetLayoutBindingFlagsCreateInfo>
            layoutChain{
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount = 1,
                    .pBindings = &layoutBinding,
                },
                vk::DescriptorSetLayoutBindingFlagsCreateInfo{
                    .bindingCount = 1,
                    .pBindingFlags = &layoutFlags,
                }};
        _dsLayouts.modelInstances =
            _device->logical().createDescriptorSetLayout(
                layoutChain.get<vk::DescriptorSetLayoutCreateInfo>());
    }

    {
        const std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{{
            {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding = 2,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute,
            },
        }};
        _dsLayouts.lights = _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = static_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });
    }

    for (auto &scene : _scenes)
    {
        {
            const std::vector<vk::DescriptorSetLayout> layouts(
                swapImageCount, _dsLayouts.modelInstances);

            scene.modelInstancesDescriptorSets =
                _device->logical().allocateDescriptorSets(
                    vk::DescriptorSetAllocateInfo{
                        .descriptorPool = _descriptorPool,
                        .descriptorSetCount =
                            static_cast<uint32_t>(layouts.size()),
                        .pSetLayouts = layouts.data(),
                    });

            std::vector<vk::DescriptorBufferInfo> infos;
            infos.reserve(scene.modelInstanceTransformsBuffers.size());
            for (auto &buffer : scene.modelInstanceTransformsBuffers)
                infos.push_back({
                    .buffer = buffer.handle,
                    .range = VK_WHOLE_SIZE,
                });

            std::vector<vk::WriteDescriptorSet> dss;
            dss.reserve(infos.size());
            for (uint32_t i = 0; i < infos.size(); ++i)
                dss.push_back(vk::WriteDescriptorSet{
                    .dstSet = scene.modelInstancesDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &infos[i],
                });

            _device->logical().updateDescriptorSets(
                static_cast<uint32_t>(dss.size()), dss.data(), 0, nullptr);
        }

        {
            const std::vector<vk::DescriptorSetLayout> layouts(
                swapImageCount, _dsLayouts.lights);

            auto &lights = scene.lights;

            lights.descriptorSets = _device->logical().allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo{
                    .descriptorPool = _descriptorPool,
                    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                    .pSetLayouts = layouts.data(),
                });

            const auto dirLightInfos = lights.directionalLight.bufferInfos();
            const auto pointLightInfos = lights.pointLights.bufferInfos();
            const auto spotLightInfos = lights.spotLights.bufferInfos();
            const auto &descriptorSets = lights.descriptorSets;
            for (size_t i = 0; i < descriptorSets.size(); ++i)
            {
                const std::array<vk::WriteDescriptorSet, 3> descriptorWrites{{
                    {
                        .dstSet = descriptorSets[i],
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .pBufferInfo = &dirLightInfos[i],
                    },
                    {
                        .dstSet = descriptorSets[i],
                        .dstBinding = 1,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &pointLightInfos[i],
                    },
                    {
                        .dstSet = descriptorSets[i],
                        .dstBinding = 2,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &spotLightInfos[i],
                    },
                }};
                _device->logical().updateDescriptorSets(
                    static_cast<uint32_t>(descriptorWrites.size()),
                    descriptorWrites.data(), 0, nullptr);
            }
        }
    }

    const std::array<vk::DescriptorSetLayoutBinding, 2> skyboxLayoutBindings{{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    }};
    _dsLayouts.skybox = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(skyboxLayoutBindings.size()),
            .pBindings = skyboxLayoutBindings.data(),
        });

    const std::vector<vk::DescriptorSetLayout> skyboxLayouts(
        swapImageCount, _dsLayouts.skybox);
    _skyboxDSs =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(skyboxLayouts.size()),
            .pSetLayouts = skyboxLayouts.data(),
        });

    const auto skyboxBufferInfos = [&]
    {
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        for (auto &buffer : _skyboxUniformBuffers)
            bufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = buffer.handle,
                .offset = 0,
                .range = sizeof(mat4),
            });
        return bufferInfos;
    }();
    const vk::DescriptorImageInfo skyboxImageInfo = _skyboxTexture.imageInfo();
    for (size_t i = 0; i < _skyboxDSs.size(); ++i)
    {
        const std::array<vk::WriteDescriptorSet, 2> writeDescriptorSets{{
            {
                .dstSet = _skyboxDSs[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &skyboxBufferInfos[i],
            },
            {
                .dstSet = _skyboxDSs[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &skyboxImageInfo,
            },
        }};
        _device->logical().updateDescriptorSets(
            static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(), 0, nullptr);
    }
}
