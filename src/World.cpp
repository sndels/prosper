#include "World.hpp"

#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <tiny_gltf.h>

#include <algorithm>
#include <iostream>
#include <set>

#include "Timer.hpp"
#include "Utils.hpp"

using namespace glm;

#ifdef _WIN32
// Windows' header doesn't include these
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_MIRRORED_REPEAT 0x8370
#endif // _WIN32 or _WIN64

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

    return device->createBuffer(BufferCreateInfo{
        .byteSize = sizeof(skyboxVerts[0]) * skyboxVerts.size(),
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .initialData = skyboxVerts.data(),
        .debugName = "SkyboxVertexBuffer",
    });
}

vk::TransformMatrixKHR convertTransform(const glm::mat4 &trfn)
{
    return vk::TransformMatrixKHR{
        .matrix = {{
            std::array<float, 4>{
                trfn[0][0], trfn[1][0], trfn[2][0], trfn[3][0]},
            std::array<float, 4>{
                trfn[0][1], trfn[1][1], trfn[2][1], trfn[3][1]},
            std::array<float, 4>{
                trfn[0][2], trfn[1][2], trfn[2][2], trfn[3][2]},
        }},
    };
}

vk::Filter getVkFilterMode(int glEnum)
{
    switch (glEnum)
    {
    case GL_NEAREST:
    case GL_NEAREST_MIPMAP_NEAREST:
    case GL_NEAREST_MIPMAP_LINEAR:
        return vk::Filter::eNearest;
    case GL_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_LINEAR_MIPMAP_LINEAR:
        return vk::Filter::eLinear;
    }

    std::cerr << "Invalid gl filter " << glEnum << std::endl;
    return vk::Filter::eLinear;
}

vk::SamplerAddressMode getVkAddressMode(int glEnum)
{
    switch (glEnum)
    {
    case GL_CLAMP_TO_EDGE:
        return vk::SamplerAddressMode::eClampToEdge;
    case GL_MIRRORED_REPEAT:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case GL_REPEAT:
        return vk::SamplerAddressMode::eRepeat;
    }
    std::cerr << "Invalid gl wrapping mode " << glEnum << std::endl;
    return vk::SamplerAddressMode::eClampToEdge;
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

    tl("BLAS creation", [&]() { createBlases(); });
    tl("TLAS creation", [&]() { createTlases(); });
    tl("Buffer creation", [&]() { createBuffers(swapImageCount); });

    createDescriptorPool(swapImageCount);
    createDescriptorSets(swapImageCount);
}

World::~World()
{
    _device->logical().destroy(_descriptorPool);

    _device->logical().destroy(_dsLayouts.lights);
    _device->logical().destroy(_dsLayouts.skybox);
    _device->logical().destroy(_dsLayouts.accelerationStructure);
    _device->logical().destroy(_dsLayouts.modelInstances);
    _device->logical().destroy(_dsLayouts.indexBuffers);
    _device->logical().destroy(_dsLayouts.vertexBuffers);
    _device->logical().destroy(_dsLayouts.materialTextures);

    _device->destroy(_skyboxVertexBuffer);
    _device->destroy(_materialsBuffer);

    for (auto &blas : _blases)
    {
        _device->logical().destroy(blas.handle);
        _device->destroy(blas.buffer);
    }
    for (auto &tlas : _tlases)
    {
        _device->logical().destroy(tlas.handle);
        _device->destroy(tlas.buffer);
    }
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
    for (auto &sampler : _samplers)
        _device->logical().destroy(sampler);
}

const Scene &World::currentScene() const { return _scenes[_currentScene]; }

void World::updateUniformBuffers(
    const Camera &cam, const uint32_t nextImage) const
{
    {
        const mat4 worldToClip =
            cam.cameraToClip() * mat4(mat3(cam.worldToCamera()));

        memcpy(
            _skyboxUniformBuffers[nextImage].mapped, &worldToClip,
            sizeof(mat4));
    }

    const auto &scene = currentScene();

    {
        std::vector<ModelInstance::Transforms> transforms;
        transforms.reserve(scene.modelInstances.size());
        for (const auto &instance : scene.modelInstances)
            transforms.push_back(instance.transforms);

        memcpy(
            scene.modelInstanceTransformsBuffers[nextImage].mapped,
            transforms.data(),
            sizeof(ModelInstance::Transforms) * transforms.size());
    }

    scene.lights.directionalLight.updateBuffer(nextImage);
    scene.lights.pointLights.updateBuffer(nextImage);
    scene.lights.spotLights.updateBuffer(nextImage);
}

void World::drawSkybox(const vk::CommandBuffer &buffer) const
{
    const vk::DeviceSize offset = 0;
    buffer.bindVertexBuffers(0, 1, &_skyboxVertexBuffer.handle, &offset);
    buffer.draw(asserted_cast<uint32_t>(SKYBOX_VERTS_SIZE), 1, 0, 0);
}

void World::loadTextures(const tinygltf::Model &gltfModel)
{
    {
        vk::SamplerCreateInfo info{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear, // TODO
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        _samplerMap.insert(std::make_pair(info, 0));
        _samplers.push_back(_device->logical().createSampler(info));
    }
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

        const vk::SamplerCreateInfo info{
            .magFilter = getVkFilterMode(sampler.magFilter),
            .minFilter = getVkFilterMode(sampler.minFilter),
            .mipmapMode = vk::SamplerMipmapMode::eLinear, // TODO
            .addressModeU = getVkAddressMode(sampler.wrapS),
            .addressModeV = getVkAddressMode(sampler.wrapT),
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = VK_TRUE, // TODO: Is there a gltf flag?
            .maxAnisotropy = 16,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        if (!_samplerMap.contains(info))
        {
            _samplerMap.insert(std::make_pair(
                info, asserted_cast<uint32_t>(_samplers.size())));
            _samplers.push_back(_device->logical().createSampler(info));
            assert(
                _samplers.size() < 0xFF &&
                "Too many samplers to pack in u32 texture index");
        }

        _textures.emplace_back(_device, image, true, _samplerMap[info]);
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
            assert(mat.baseColor < 0x00FFFFFF && "Sampler index won't fit");
            mat.baseColor |= _textures[mat.baseColor - 1].sampler << 24;
        }
        if (const auto &elem = material.values.find("metallicRoughnessTexture");
            elem != material.values.end())
        {
            mat.metallicRoughness = elem->second.TextureIndex() + 1;
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Metallic roughness TexCoord isn't 0\n",
                    material.name.c_str());
            assert(
                mat.metallicRoughness < 0x00FFFFFF &&
                "Sampler index won't fit");
            mat.metallicRoughness |=
                _textures[mat.metallicRoughness - 1].sampler << 24;
        }
        if (const auto &elem = material.additionalValues.find("normalTexture");
            elem != material.additionalValues.end())
        {
            mat.normal = elem->second.TextureIndex() + 1;
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Normal TexCoord isn't 0\n",
                    material.name.c_str());
            assert(mat.normal < 0x00FFFFFF && "Sampler index won't fit");
            mat.normal |= _textures[mat.normal - 1].sampler << 24;
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
        _models.push_back({});
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
                        .pos = vec3{make_vec3(&positions[v * 3])},
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

            _meshes.emplace_back(_device, vertices, indices);
            _models.back().subModels.push_back({
                .meshID = asserted_cast<uint32_t>(_meshes.size() - 1),
                .materialID = material,
            });
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
            _nodes[n].modelID = node.mesh;
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

            lights[&_nodes[n]] = asserted_cast<size_t>(light.GetNumberAsInt());
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

                const auto normalToWorld = transpose(inverse(modelToWorld));
                if (node->modelID != 0xFFFFFFFF)
                {
                    scene.modelInstances.push_back(
                        {.id = asserted_cast<uint32_t>(
                             scene.modelInstances.size()),
                         .modelID = node->modelID,
                         .transforms = {
                             .modelToWorld = modelToWorld,
                             .normalToWorld = normalToWorld,
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

void World::createBlases()
{
    _blases.resize(_meshes.size());
    for (size_t i = 0; i < _blases.size(); ++i)
    {
        const auto &mesh = _meshes[i];
        auto &blas = _blases[i];
        // Basics from RT Gems II chapter 16

        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData =
                _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                    .buffer = mesh.vertexBuffer(),
                }),
            .vertexStride = sizeof(Vertex),
            .maxVertex = _meshes[i].vertexCount(),
            .indexType = vk::IndexType::eUint32,
            .indexData =
                _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                    .buffer = mesh.indexBuffer(),
                }),
        };
        const vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = triangles,
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = mesh.indexCount() / 3,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        };
        // dst and scratch will be set once allocated
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags =
                vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };

        // TODO: This stuff is ~the same for TLAS and BLAS
        const auto sizeInfo =
            _device->logical().getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
                {rangeInfo.primitiveCount});

        blas.buffer = _device->createBuffer(BufferCreateInfo{
            .byteSize = sizeInfo.accelerationStructureSize,
            .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            .debugName = "BLASBuffer",
        });

        const vk::AccelerationStructureCreateInfoKHR createInfo{
            .buffer = blas.buffer.handle,
            .size = sizeInfo.accelerationStructureSize,
            .type = buildInfo.type,
        };
        blas.handle =
            _device->logical().createAccelerationStructureKHR(createInfo);

        buildInfo.dstAccelerationStructure = blas.handle;

        // TODO: Reuse and grow scratch
        const auto scratchBuffer = _device->createBuffer(BufferCreateInfo{
            .byteSize = sizeInfo.buildScratchSize,
            .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            .debugName = "ScratchBuffer",
        });

        buildInfo.scratchData =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = scratchBuffer.handle,
            });

        const auto cb = _device->beginGraphicsCommands();

        const auto *pRangeInfo = &rangeInfo;
        // TODO: Build multiple blas at a time/with the same cb
        cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

        _device->endGraphicsCommands(cb);

        _device->destroy(scratchBuffer);
    }
}

void World::createTlases()
{
    _tlases.resize(_scenes.size());
    for (size_t i = 0; i < _tlases.size(); ++i)
    {
        const auto &scene = _scenes[i];
        auto &tlas = _tlases[i];
        // Basics from RT Gems II chapter 16

        std::vector<vk::AccelerationStructureInstanceKHR> instances;
        std::vector<std::tuple<const Model &, vk::TransformMatrixKHR>>
            modelInstances;
        modelInstances.reserve(scene.modelInstances.size());

        size_t instanceCount = 0;
        for (const auto &mi : scene.modelInstances)
        {
            const auto &model = _models[mi.modelID];
            modelInstances.emplace_back(
                model, convertTransform(mi.transforms.modelToWorld));
            instanceCount += model.subModels.size();
        }
        instances.reserve(instanceCount);

        for (const auto &[model, trfn] : modelInstances)
        {
            for (const auto &sm : model.subModels)
            {
                assert(sm.meshID <= 0x7FFF);
                assert(sm.materialID <= 0x1FF);

                const auto &blas = _blases[sm.meshID];
                assert(blas.handle != vk::AccelerationStructureKHR{});

                instances.push_back(vk::AccelerationStructureInstanceKHR{
                    .transform = trfn,
                    .instanceCustomIndex = (sm.meshID << 9) | sm.materialID,
                    .mask = 0xFF,
                    .accelerationStructureReference =
                        _device->logical().getAccelerationStructureAddressKHR(
                            vk::AccelerationStructureDeviceAddressInfoKHR{
                                .accelerationStructure = blas.handle,
                            }),
                });
            }
        }

        auto instancesBuffer = _device->createBuffer(BufferCreateInfo{
            .byteSize = sizeof(instances[0]) * instances.size(),
            .usage = vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::
                         eAccelerationStructureBuildInputReadOnlyKHR,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            .initialData = instances.data(),
            .debugName = "InstancesBuffer",
        });

        // Need a barrier here if a shared command buffer is used so that the
        // copy happens before the build

        const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = asserted_cast<uint32_t>(instances.size()),
            .primitiveOffset = 0,
        };

        const vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
            .data =
                _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                    .buffer = instancesBuffer.handle,
                }),
        };
        const vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = instancesData,
        };
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };

        // TODO: This stuff is ~the same for TLAS and BLAS
        const auto sizeInfo =
            _device->logical().getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
                {rangeInfo.primitiveCount});

        tlas.buffer = _device->createBuffer(BufferCreateInfo{
            .byteSize = sizeInfo.accelerationStructureSize,
            .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            .debugName = "TLASBuffer",
        });

        const vk::AccelerationStructureCreateInfoKHR createInfo{
            .buffer = tlas.buffer.handle,
            .size = sizeInfo.accelerationStructureSize,
            .type = buildInfo.type,
        };
        tlas.handle =
            _device->logical().createAccelerationStructureKHR(createInfo);

        buildInfo.dstAccelerationStructure = tlas.handle;

        // TODO: Reuse and grow scratch
        const auto scratchBuffer = _device->createBuffer(BufferCreateInfo{
            .byteSize = sizeInfo.buildScratchSize,
            .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            .debugName = "ScratchBuffer",
        });

        buildInfo.scratchData =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = scratchBuffer.handle,
            });

        const auto cb = _device->beginGraphicsCommands();

        const auto *pRangeInfo = &rangeInfo;
        // TODO: Use a single cb for instance buffer copies and builds for all
        //       tlases need a barrier after buffer copy and build!
        cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

        _device->endGraphicsCommands(cb);

        _device->destroy(scratchBuffer);
        _device->destroy(instancesBuffer);
    }
}

void World::createBuffers(const uint32_t swapImageCount)
{
    _materialsBuffer = _device->createBuffer(BufferCreateInfo{
        .byteSize = _materials.size() * sizeof(_materials[0]),
        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eTransferDst,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .initialData = _materials.data(),
        .debugName = "MaterialsBuffer",
    });

    {
        for (auto &scene : _scenes)
        {
            {
                const vk::DeviceSize bufferSize =
                    sizeof(ModelInstance::Transforms) *
                    scene.modelInstances.size();
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.modelInstanceTransformsBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .byteSize = bufferSize,
                            .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                            .properties =
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            .createMapped = true,
                            .debugName = "InstanceTransforms",
                        }));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(DirectionalLight::Parameters);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.directionalLight.uniformBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .byteSize = bufferSize,
                            .usage = vk::BufferUsageFlagBits::eUniformBuffer,
                            .properties =
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            .createMapped = true,
                            .debugName = "DirectionalLightUniforms",
                        }));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(PointLights::BufferData);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.pointLights.storageBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .byteSize = bufferSize,
                            .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                            .properties =
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            .createMapped = true,
                            .debugName = "PointLightsBuffer",
                        }));
            }

            {
                const vk::DeviceSize bufferSize =
                    sizeof(SpotLights::BufferData);
                for (size_t i = 0; i < swapImageCount; ++i)
                    scene.lights.spotLights.storageBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .byteSize = bufferSize,
                            .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                            .properties =
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                            .createMapped = true,
                            .debugName = "SpotLightsBuffer",
                        }));
            }
        }
    }

    {
        const vk::DeviceSize bufferSize = sizeof(mat4);
        for (size_t i = 0; i < swapImageCount; ++i)
        {
            _skyboxUniformBuffers.push_back(
                _device->createBuffer(BufferCreateInfo{
                    .byteSize = bufferSize,
                    .usage = vk::BufferUsageFlagBits::eUniformBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                    .createMapped = true,
                    .debugName = "SkyboxUniforms" + std::to_string(i),
                }));
        }
    }
}

void World::createDescriptorPool(const uint32_t swapImageCount)
{
    // TODO:
    // Pool sizes and max set's don't seem to make sense. Re-read and rework.

    // TODO: Tight bound for node descriptor count by nodes with a mesh
    // Skybox cubemap is also one descriptor per image
    // As is the directional light
    const uint32_t uniformDescriptorCount =
        swapImageCount * (asserted_cast<uint32_t>(_nodes.size()) + 2);
    const uint32_t samplerDescriptorCount = 1; // TODO: Actual coutn
    const uint32_t sampledImageDescriptorCount =
        3 * asserted_cast<uint32_t>(_materials.size()) + swapImageCount;
    const std::array<vk::DescriptorPoolSize, 4> poolSizes{
        {{// Dynamic need per frame descriptor sets of one descriptor per UBlock
          vk::DescriptorType::eUniformBuffer, uniformDescriptorCount},
         {// Samplers need one descriptor per texture as they are constant
          // between frames
          vk::DescriptorType::eSampler, samplerDescriptorCount},
         {// Materials need one descriptor per texture as they are constant
          // between frames
          vk::DescriptorType::eSampledImage, sampledImageDescriptorCount},
         {// Lights require per frame descriptors for points, spots
          vk::DescriptorType::eStorageBuffer, 2 * swapImageCount}},
    };
    // Per-frame: Nodes, skybox, dirlight, points and spots
    // Single: Materials
    const uint32_t maxSets =
        swapImageCount * ((asserted_cast<uint32_t>(_nodes.size()) + 3)) +
        asserted_cast<uint32_t>(_materials.size());
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets,
            .poolSizeCount = asserted_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });
}

void World::createDescriptorSets(const uint32_t swapImageCount)
{
    if (_device == nullptr)
        throw std::runtime_error(
            "Tried to create World descriptor sets before loading glTF");

    std::vector<vk::WriteDescriptorSet> dss;

    // Materials layout and descriptors set
    // Define outside the helper scope to keep alive until updateDescriptorSets
    const vk::DescriptorBufferInfo materialDatasInfo{
        .buffer = _materialsBuffer.handle,
        .range = VK_WHOLE_SIZE,
    };
    std::vector<vk::DescriptorImageInfo> materialSamplerInfos;
    std::vector<vk::DescriptorImageInfo> materialImageInfos;
    {
        materialSamplerInfos.reserve(_samplers.size());
        for (const auto &s : _samplers)
            materialSamplerInfos.push_back(
                vk::DescriptorImageInfo{.sampler = s});
        const auto samplerInfoCount =
            asserted_cast<uint32_t>(materialSamplerInfos.size());
        _dsLayouts.materialSamplerCount = samplerInfoCount;

        materialImageInfos.reserve(_textures.size() + 1);
        materialImageInfos.push_back(_emptyTexture.imageInfo());
        for (const auto &[tex, _] : _textures)
            materialImageInfos.push_back(tex.imageInfo());
        const auto imageInfoCount =
            asserted_cast<uint32_t>(materialImageInfos.size());

        const std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = samplerInfoCount,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1 + samplerInfoCount,
                .descriptorType = vk::DescriptorType::eSampledImage,
                .descriptorCount = imageInfoCount,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        const std::array<vk::DescriptorBindingFlags, 3> layoutFlags{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlagBits::eVariableDescriptorCount,
        };
        const vk::StructureChain<
            vk::DescriptorSetLayoutCreateInfo,
            vk::DescriptorSetLayoutBindingFlagsCreateInfo>
            layoutChain{
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount =
                        asserted_cast<uint32_t>(layoutBindings.size()),
                    .pBindings = layoutBindings.data(),
                },
                vk::DescriptorSetLayoutBindingFlagsCreateInfo{
                    .bindingCount = asserted_cast<uint32_t>(layoutFlags.size()),
                    .pBindingFlags = layoutFlags.data(),
                }};
        _dsLayouts.materialTextures =
            _device->logical().createDescriptorSetLayout(
                layoutChain.get<vk::DescriptorSetLayoutCreateInfo>());

        const vk::StructureChain<
            vk::DescriptorSetAllocateInfo,
            vk::DescriptorSetVariableDescriptorCountAllocateInfo>
            dsChain{
                vk::DescriptorSetAllocateInfo{
                    .descriptorPool = _descriptorPool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &_dsLayouts.materialTextures},
                vk::DescriptorSetVariableDescriptorCountAllocateInfo{
                    .descriptorSetCount = 1,
                    .pDescriptorCounts = &imageInfoCount,
                }};
        _materialTexturesDS = _device->logical().allocateDescriptorSets(
            dsChain.get<vk::DescriptorSetAllocateInfo>())[0];

        dss.reserve(dss.size() + 3);
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _materialTexturesDS,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &materialDatasInfo,
        });
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _materialTexturesDS,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount =
                asserted_cast<uint32_t>(materialSamplerInfos.size()),
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = materialSamplerInfos.data(),
        });
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _materialTexturesDS,
            .dstBinding =
                1 + asserted_cast<uint32_t>(materialSamplerInfos.size()),
            .dstArrayElement = 0,
            .descriptorCount =
                asserted_cast<uint32_t>(materialImageInfos.size()),
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = materialImageInfos.data(),
        });
    }

    // Geometry layouts and descriptor set
    // Define outside the helper scope to keep alive until updateDescriptorSets
    std::vector<vk::DescriptorBufferInfo> vertexBufferInfos;
    std::vector<vk::DescriptorBufferInfo> indexBufferInfos;
    {
        vertexBufferInfos.reserve(_meshes.size());
        vertexBufferInfos.reserve(_meshes.size());
        for (const auto &m : _meshes)
        {
            vertexBufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = m.vertexBuffer(),
                .range = VK_WHOLE_SIZE,
            });
            indexBufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = m.indexBuffer(),
                .range = VK_WHOLE_SIZE,
            });
        }
        const auto vertexBuffersCount =
            asserted_cast<uint32_t>(vertexBufferInfos.size());
        const auto indexBuffersCount =
            asserted_cast<uint32_t>(indexBufferInfos.size());

        const auto createDSLayout =
            [this](vk::DescriptorSetLayout *layout, uint32_t descriptorCount)
        {
            const vk::DescriptorSetLayoutBinding layoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = descriptorCount,
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
            };
            const vk::DescriptorBindingFlags variableDescriptorsFlag =
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
            const vk::StructureChain<
                vk::DescriptorSetLayoutCreateInfo,
                vk::DescriptorSetLayoutBindingFlagsCreateInfo>
                layoutChain{
                    vk::DescriptorSetLayoutCreateInfo{
                        .bindingCount = 1,
                        .pBindings = &layoutBinding,
                    },
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo{
                        .bindingCount = 1,
                        .pBindingFlags = &variableDescriptorsFlag,
                    }};
            *layout = _device->logical().createDescriptorSetLayout(
                layoutChain.get<vk::DescriptorSetLayoutCreateInfo>());
        };
        createDSLayout(&_dsLayouts.vertexBuffers, vertexBuffersCount);
        createDSLayout(&_dsLayouts.indexBuffers, indexBuffersCount);

        const auto createDS = [this](
                                  vk::DescriptorSet *ds,
                                  vk::DescriptorSetLayout layout,
                                  uint32_t descriptorCount)
        {
            const vk::StructureChain<
                vk::DescriptorSetAllocateInfo,
                vk::DescriptorSetVariableDescriptorCountAllocateInfo>
                dsChain{
                    vk::DescriptorSetAllocateInfo{
                        .descriptorPool = _descriptorPool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &layout},
                    vk::DescriptorSetVariableDescriptorCountAllocateInfo{
                        .descriptorSetCount = 1,
                        .pDescriptorCounts = &descriptorCount,
                    }};
            *ds = _device->logical().allocateDescriptorSets(
                dsChain.get<vk::DescriptorSetAllocateInfo>())[0];
        };
        createDS(
            &_vertexBuffersDS, _dsLayouts.vertexBuffers, vertexBuffersCount);
        createDS(&_indexBuffersDS, _dsLayouts.indexBuffers, indexBuffersCount);

        dss.reserve(dss.size() + 2);
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _vertexBuffersDS,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount =
                asserted_cast<uint32_t>(vertexBufferInfos.size()),
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = vertexBufferInfos.data(),
        });
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _indexBuffersDS,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = asserted_cast<uint32_t>(indexBufferInfos.size()),
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = indexBufferInfos.data(),
        });
    }

    // Acceleration structure layout
    {
        const vk::DescriptorSetLayoutBinding layoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
        };
        const vk::DescriptorSetLayoutCreateInfo createInfo{
            .bindingCount = 1,
            .pBindings = &layoutBinding,
        };
        _dsLayouts.accelerationStructure =
            _device->logical().createDescriptorSetLayout(createInfo);
    }

    // Model instances layout
    {
        const vk::DescriptorSetLayoutBinding layoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };
        const vk::DescriptorBindingFlags layoutFlags{};
        const vk::StructureChain<
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

    // Lights layout
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
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });
    }

    // Scene descriptor sets
    // Define outside the helper scope to keep alive until updateDescriptorSets
    std::vector<vk::DescriptorBufferInfo> modelInstanceInfos;
    std::vector<vk::DescriptorBufferInfo> lightInfos;
    std::vector<vk::StructureChain<
        vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR>>
        asDSChains;
    {
        size_t sceneI = 0;
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
                                asserted_cast<uint32_t>(layouts.size()),
                            .pSetLayouts = layouts.data(),
                        });

                modelInstanceInfos.reserve(
                    modelInstanceInfos.size() +
                    scene.modelInstanceTransformsBuffers.size());
                const auto startIndex =
                    asserted_cast<uint32_t>(modelInstanceInfos.size());
                for (auto &buffer : scene.modelInstanceTransformsBuffers)
                    modelInstanceInfos.push_back({
                        .buffer = buffer.handle,
                        .range = VK_WHOLE_SIZE,
                    });

                dss.reserve(
                    dss.size() + modelInstanceInfos.size() - startIndex);
                for (uint32_t i = startIndex; i < modelInstanceInfos.size();
                     ++i)
                    dss.push_back(vk::WriteDescriptorSet{
                        .dstSet = scene.modelInstancesDescriptorSets[i],
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &modelInstanceInfos[i],
                    });
            }

            {
                const std::vector<vk::DescriptorSetLayout> layouts(
                    swapImageCount, _dsLayouts.lights);

                auto &lights = scene.lights;

                lights.descriptorSets =
                    _device->logical().allocateDescriptorSets(
                        vk::DescriptorSetAllocateInfo{
                            .descriptorPool = _descriptorPool,
                            .descriptorSetCount =
                                asserted_cast<uint32_t>(layouts.size()),
                            .pSetLayouts = layouts.data(),
                        });

                const auto dirLightInfos =
                    lights.directionalLight.bufferInfos();
                const auto pointLightInfos = lights.pointLights.bufferInfos();
                const auto spotLightInfos = lights.spotLights.bufferInfos();

                const auto dirLightStart = lightInfos.size();
                const auto pointLightStart =
                    dirLightStart + dirLightInfos.size();
                const auto spotLightStart =
                    pointLightStart + pointLightInfos.size();

                lightInfos.reserve(
                    lightInfos.size() + dirLightInfos.size() +
                    pointLightInfos.size() + spotLightInfos.size());
                lightInfos.insert(
                    lightInfos.end(), dirLightInfos.begin(),
                    dirLightInfos.end());
                lightInfos.insert(
                    lightInfos.end(), pointLightInfos.begin(),
                    pointLightInfos.end());
                lightInfos.insert(
                    lightInfos.end(), spotLightInfos.begin(),
                    spotLightInfos.end());

                const auto &descriptorSets = lights.descriptorSets;
                dss.reserve(dss.size() + descriptorSets.size() * 3);
                for (size_t i = 0; i < descriptorSets.size(); ++i)
                {
                    dss.push_back(vk::WriteDescriptorSet{
                        .dstSet = descriptorSets[i],
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .pBufferInfo = &lightInfos[dirLightStart + i],
                    });
                    dss.push_back(vk::WriteDescriptorSet{
                        .dstSet = descriptorSets[i],
                        .dstBinding = 1,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &lightInfos[pointLightStart + i],
                    });
                    dss.push_back(vk::WriteDescriptorSet{
                        .dstSet = descriptorSets[i],
                        .dstBinding = 2,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &lightInfos[spotLightStart + i],
                    });
                }
            }

            {
                // TODO: DS per frame when TLAS is updated
                scene.accelerationStructureDS =
                    _device->logical().allocateDescriptorSets(
                        vk::DescriptorSetAllocateInfo{
                            .descriptorPool = _descriptorPool,
                            .descriptorSetCount = 1,
                            .pSetLayouts = &_dsLayouts.accelerationStructure,
                        })[0];

                asDSChains.emplace_back(
                    vk::WriteDescriptorSet{
                        .dstSet = scene.accelerationStructureDS,
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType =
                            vk::DescriptorType::eAccelerationStructureKHR,
                    },
                    vk::WriteDescriptorSetAccelerationStructureKHR{
                        .accelerationStructureCount = 1,
                        .pAccelerationStructures = &_tlases[0].handle,
                    });

                dss.push_back(asDSChains.back().get<vk::WriteDescriptorSet>());
            }
            sceneI++;
        }
    }

    // Skybox layout and descriptor sets
    std::vector<vk::DescriptorBufferInfo> skyboxBufferInfos;
    vk::DescriptorImageInfo skyboxImageInfo;
    {
        const std::array<vk::DescriptorSetLayoutBinding, 2>
            skyboxLayoutBindings{{
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
                .bindingCount =
                    asserted_cast<uint32_t>(skyboxLayoutBindings.size()),
                .pBindings = skyboxLayoutBindings.data(),
            });

        const std::vector<vk::DescriptorSetLayout> skyboxLayouts(
            swapImageCount, _dsLayouts.skybox);
        _skyboxDSs = _device->logical().allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{
                .descriptorPool = _descriptorPool,
                .descriptorSetCount =
                    asserted_cast<uint32_t>(skyboxLayouts.size()),
                .pSetLayouts = skyboxLayouts.data(),
            });

        skyboxBufferInfos.reserve(_skyboxUniformBuffers.size());
        for (auto &buffer : _skyboxUniformBuffers)
            skyboxBufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = buffer.handle,
                .offset = 0,
                .range = sizeof(mat4),
            });
        skyboxImageInfo = _skyboxTexture.imageInfo();

        dss.reserve(dss.size() + _skyboxDSs.size() * 2);
        for (size_t i = 0; i < _skyboxDSs.size(); ++i)
        {
            dss.push_back(vk::WriteDescriptorSet{
                .dstSet = _skyboxDSs[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &skyboxBufferInfos[i],
            });
            dss.push_back(vk::WriteDescriptorSet{
                .dstSet = _skyboxDSs[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &skyboxImageInfo,
            });
        }
    }

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(dss.size()), dss.data(), 0, nullptr);
}
