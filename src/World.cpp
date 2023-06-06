#include "World.hpp"

#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <cstdlib>
#include <iostream>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/pair.hpp>
#include <wheels/containers/static_array.hpp>

#include "Timer.hpp"
#include "Utils.hpp"

using namespace glm;
using namespace wheels;

#ifdef _WIN32
// Windows' header doesn't include these
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_MIRRORED_REPEAT 0x8370
#endif // _WIN32 or _WIN64

namespace
{

// A GB should be plenty for the types of scenes I use, but not so much that
// a reasonable device wouldn't have enough available
constexpr size_t sWorldMemSize = megabytes(1024);

constexpr size_t SKYBOX_VERTS_SIZE = 36;

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
    assert(device != nullptr);

    // Avoid large global allocation
    const StaticArray<glm::vec3, SKYBOX_VERTS_SIZE> skyboxVerts{
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
        .desc =
            BufferDescription{
                .byteSize = sizeof(skyboxVerts[0]) * skyboxVerts.size(),
                .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .initialData = skyboxVerts.data(),
        .debugName = "SkyboxVertexBuffer",
    });
}

constexpr vk::TransformMatrixKHR convertTransform(const glm::mat4 &trfn)
{
    return vk::TransformMatrixKHR{
        .matrix = {{
            std::array{trfn[0][0], trfn[1][0], trfn[2][0], trfn[3][0]},
            std::array{trfn[0][1], trfn[1][1], trfn[2][1], trfn[3][1]},
            std::array{trfn[0][2], trfn[1][2], trfn[2][2], trfn[3][2]},
        }},
    };
}

constexpr vk::Filter getVkFilterMode(int glEnum)
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

constexpr vk::SamplerAddressMode getVkAddressMode(int glEnum)
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

Buffer createTextureStaging(Device *device)
{
    // Assume at most 4k at 8bits per channel
    const vk::DeviceSize stagingSize = 4096 * 4096 * sizeof(uint32_t);
    return device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = stagingSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });
}

} // namespace

World::World(
    ScopedScratch scopeAlloc, Device *device,
    const std::filesystem::path &scene, bool deferredLoading)
: _linearAlloc{sWorldMemSize}
, _sceneDir{resPath(scene.parent_path())}
, _skyboxTexture{scopeAlloc.child_scope(), device, resPath("env/storm.ktx")}
, _skyboxVertexBuffer{createSkyboxVertexBuffer(device)}
, _device{device}
// Use general for descriptors because because we don't know the required
// storage up front and the internal array will be reallocated
, _descriptorAllocator{_generalAlloc, device}

{
    assert(_device != nullptr);

    printf("Loading world\n");

    Timer t;
    const auto gltfModel = loadGLTFModel(resPath(scene));
    printf("glTF model loading took %.2fs\n", t.getSeconds());

    if (deferredLoading)
        _deferredLoadingContext.emplace(_generalAlloc, _device, gltfModel);

    const auto &tl = [&](const char *stage, std::function<void()> const &fn)
    {
        t.reset();
        fn();
        printf("%s took %.2fs\n", stage, t.getSeconds());
    };

    Array<Texture2DSampler> texture2DSamplers{
        _generalAlloc, gltfModel.textures.size() + 1};
    tl("Texture loading",
       [&]()
       {
           loadTextures(
               scopeAlloc.child_scope(), gltfModel, texture2DSamplers,
               deferredLoading);
       });
    tl("Material loading",
       [&]() { loadMaterials(gltfModel, texture2DSamplers, deferredLoading); });
    tl("Model loading ", [&]() { loadModels(gltfModel); });
    tl("Scene loading ",
       [&]() { loadScenes(scopeAlloc.child_scope(), gltfModel); });

    tl("BLAS creation", [&]() { createBlases(); });
    tl("TLAS creation", [&]() { createTlases(scopeAlloc.child_scope()); });
    tl("Buffer creation", [&]() { createBuffers(); });

    createDescriptorSets(scopeAlloc.child_scope());
}

World::~World()
{
    _device->logical().destroy(_dsLayouts.lights);
    _device->logical().destroy(_dsLayouts.skybox);
    _device->logical().destroy(_dsLayouts.skyboxOnly);
    _device->logical().destroy(_dsLayouts.rayTracing);
    _device->logical().destroy(_dsLayouts.modelInstances);
    _device->logical().destroy(_dsLayouts.geometry);
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

        _device->destroy(scene.rtInstancesBuffer);

        for (auto &buffer : scene.lights.directionalLight.uniformBuffers)
            _device->destroy(buffer);

        for (auto &buffer : scene.lights.pointLights.storageBuffers)
            _device->destroy(buffer);

        for (auto &buffer : scene.lights.spotLights.storageBuffers)
            _device->destroy(buffer);
    }
    for (auto &buffer : _geometryBuffers)
        _device->destroy(buffer);
    _device->destroy(_meshBuffersBuffer);
    for (auto &buffer : _skyboxUniformBuffers)
        _device->destroy(buffer);
    for (auto &sampler : _samplers)
        _device->logical().destroy(sampler);
}

void World::handleDeferredLoading(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame,
    Profiler &profiler)
{
    if (!_deferredLoadingContext.has_value())
        return;

    if (_deferredLoadingContext->loadedImageCount ==
        _deferredLoadingContext->gltfModel.images.size())
    {
        // Don't clean up until all in flight uploads are finished
        if (_deferredLoadingContext->framesSinceFinish++ > MAX_FRAMES_IN_FLIGHT)
            _deferredLoadingContext.reset();
        return;
    }

    // No gpu as timestamps are flaky for this work
    const auto _s = profiler.createCpuScope("DeferredLoading");

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    assert(ctx.loadedImageCount < ctx.gltfModel.images.size());

    const tinygltf::Image &image = ctx.gltfModel.images[ctx.loadedImageCount];
    if (image.uri.empty())
        _texture2Ds.emplace_back(
            scopeAlloc.child_scope(), _device, image, cb,
            ctx.stagingBuffers[nextFrame], true);
    else
        _texture2Ds.emplace_back(
            _device, _sceneDir / image.uri, cb, ctx.stagingBuffers[nextFrame],
            true);

    const vk::DescriptorImageInfo imageInfo = _texture2Ds.back().imageInfo();
    const vk::WriteDescriptorSet descriptorWrite{
        .dstSet = _materialTexturesDS,
        .dstBinding = ctx.textureArrayBinding,
        // loadedImageCount is gltf images so bump by one to take our default
        // texture into account
        .dstArrayElement = ctx.loadedImageCount + 1,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eSampledImage,
        .pImageInfo = &imageInfo,
    };
    _device->logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);

    ctx.loadedImageCount++;

    // TODO:
    // Update material texture indices for remaining mats until unloaded texture
    // found Also change reset condition to check loaded material count once
    // implemented
}

const Scene &World::currentScene() const { return _scenes[_currentScene]; }

void World::updateUniformBuffers(
    const Camera &cam, const uint32_t nextFrame, ScopedScratch scopeAlloc) const
{
    {
        const mat4 worldToClip =
            cam.cameraToClip() * mat4(mat3(cam.worldToCamera()));

        memcpy(
            _skyboxUniformBuffers[nextFrame].mapped, &worldToClip,
            sizeof(mat4));
    }

    const auto &scene = currentScene();

    {
        Array<Scene::RTInstance> rtInstances{scopeAlloc, scene.rtInstanceCount};
        Array<ModelInstance::Transforms> transforms{
            scopeAlloc, scene.modelInstances.size()};

        // The RTInstances generated here have to match the indices that get
        // assigned to tlas instances
        for (auto mi = 0u; mi < scene.modelInstances.size(); ++mi)
        {
            const auto &instance = scene.modelInstances[mi];
            transforms.push_back(instance.transforms);
            for (const auto &model : _models[instance.modelID].subModels)
            {
                rtInstances.push_back(Scene::RTInstance{
                    .modelInstanceID = mi,
                    .meshID = model.meshID,
                    .materialID = model.materialID,
                });
            }
        }

        memcpy(
            scene.modelInstanceTransformsBuffers[nextFrame].mapped,
            transforms.data(),
            sizeof(ModelInstance::Transforms) * transforms.size());

        memcpy(
            scene.rtInstancesBuffer.mapped, rtInstances.data(),
            sizeof(Scene::RTInstance) * rtInstances.size());
    }

    scene.lights.directionalLight.updateBuffer(nextFrame);
    scene.lights.pointLights.updateBuffer(nextFrame);
    scene.lights.spotLights.updateBuffer(nextFrame);
}

void World::drawSkybox(const vk::CommandBuffer &buffer) const
{
    const vk::DeviceSize offset = 0;
    buffer.bindVertexBuffers(0, 1, &_skyboxVertexBuffer.handle, &offset);
    buffer.draw(asserted_cast<uint32_t>(SKYBOX_VERTS_SIZE), 1, 0, 0);
}

void World::loadTextures(
    ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
    Array<Texture2DSampler> &texture2DSamplers, bool deferredLoading)
{
    {
        const vk::SamplerCreateInfo info{
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
        _samplers.push_back(_device->logical().createSampler(info));
    }
    assert(
        gltfModel.samplers.size() < 0xFE &&
        "Too many samplers to pack in u32 texture index");
    for (const auto &sampler : gltfModel.samplers)
    {
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
        _samplers.push_back(_device->logical().createSampler(info));
    }

    const Buffer stagingBuffer = createTextureStaging(_device);

    _texture2Ds.reserve(gltfModel.images.size() + 1);
    {
        const vk::CommandBuffer cb = _device->beginGraphicsCommands();
        _texture2Ds.emplace_back(
            _device, resPath("texture/empty.png"), cb, stagingBuffer, false);
        _device->endGraphicsCommands(cb);

        texture2DSamplers.emplace_back();
    }

    assert(
        gltfModel.images.size() < 0xFFFFFE &&
        "Too many textures to pack in u32 texture index");
    if (!deferredLoading)
    {
        for (const auto &image : gltfModel.images)
        {
            const vk::CommandBuffer cb = _device->beginGraphicsCommands();
            if (image.uri.empty())
                _texture2Ds.emplace_back(
                    scopeAlloc.child_scope(), _device, image, cb, stagingBuffer,
                    true);
            else
                _texture2Ds.emplace_back(
                    _device, _sceneDir / image.uri, cb, stagingBuffer, true);
            _device->endGraphicsCommands(cb);
        }
    }

    _device->destroy(stagingBuffer);

    for (const auto &texture : gltfModel.textures)
        texture2DSamplers.emplace_back(
            asserted_cast<uint32_t>(texture.source + 1),
            asserted_cast<uint32_t>(texture.sampler + 1));
}

void World::loadMaterials(
    const tinygltf::Model &gltfModel,
    const Array<Texture2DSampler> &texture2DSamplers, bool deferredLoading)
{
    _materials.push_back(Material{});

    for (const auto &material : gltfModel.materials)
    {
        Material mat;
        if (const auto &elem = material.values.find("baseColorTexture");
            elem != material.values.end())
        {
            mat.baseColor = texture2DSamplers[elem->second.TextureIndex() + 1];
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Base color TexCoord isn't 0\n",
                    material.name.c_str());
        }
        if (const auto &elem = material.values.find("metallicRoughnessTexture");
            elem != material.values.end())
        {
            mat.metallicRoughness =
                texture2DSamplers[elem->second.TextureIndex() + 1];
            if (elem->second.TextureTexCoord() != 0)
                fprintf(
                    stderr, "%s: Metallic roughness TexCoord isn't 0\n",
                    material.name.c_str());
        }
        if (const auto &elem = material.additionalValues.find("normalTexture");
            elem != material.additionalValues.end())
        {
            mat.normal = texture2DSamplers[elem->second.TextureIndex() + 1];
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

        if (deferredLoading)
        {
            assert(_deferredLoadingContext.has_value());
            // Copy the alpha mode of the real material because that's used to
            // set opaque flag in rt
            _materials.push_back(Material{
                .alphaMode = mat.alphaMode,
            });
            _deferredLoadingContext->materials.push_back(mat);
        }
        else
            _materials.push_back(mat);
    }
}

void World::loadModels(const tinygltf::Model &gltfModel)
{
    for (const auto &b : gltfModel.buffers)
        _geometryBuffers.push_back(_device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = asserted_cast<uint32_t>(b.data.size()),
                    .usage = vk::BufferUsageFlagBits::
                                 eAccelerationStructureBuildInputReadOnlyKHR |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .initialData = b.data.data(),
            .debugName = "GeometryBuffer",
        }));

    _models.reserve(gltfModel.meshes.size());

    size_t totalPrimitiveCount = 0;
    for (const auto &mesh : gltfModel.meshes)
        totalPrimitiveCount += mesh.primitives.size();
    _meshBuffers.reserve(totalPrimitiveCount);
    _meshInfos.reserve(totalPrimitiveCount);

    for (const auto &mesh : gltfModel.meshes)
    {
        _models.emplace_back(_linearAlloc);
        Model &model = _models.back();
        model.subModels.reserve(mesh.primitives.size());
        for (const auto &primitive : mesh.primitives)
        {
            auto assertedGetAttr =
                [&](const std::string &name,
                    bool shouldHave =
                        false) -> Pair<MeshBuffers::Buffer, uint32_t>
            {
                const auto &attribute = primitive.attributes.find(name);
                if (attribute == primitive.attributes.end())
                {
                    if (shouldHave)
                        throw std::runtime_error(
                            "Primitive attribute '" + name + "' missing");
                    return make_pair(MeshBuffers::Buffer{}, 0u);
                }

                const auto &accessor = gltfModel.accessors[attribute->second];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto offset = asserted_cast<uint32_t>(
                    accessor.byteOffset + view.byteOffset);
                assert(
                    offset % sizeof(uint32_t) == 0 &&
                    "Shader binds buffers as uint");

                return make_pair(
                    MeshBuffers::Buffer{
                        .index = asserted_cast<uint32_t>(view.buffer),
                        .offset =
                            offset / static_cast<uint32_t>(sizeof(uint32_t)),
                    },
                    asserted_cast<uint32_t>(accessor.count));
            };

            // Retrieve attribute buffers
            const auto [positions, positionsCount] =
                assertedGetAttr("POSITION", true);
            const auto [normals, normalsCount] =
                assertedGetAttr("NORMAL", true);
            const auto [tangents, tangentsCount] = assertedGetAttr("TANGENT");
            const auto [texCoord0s, texCoord0sCount] =
                assertedGetAttr("TEXCOORD_0");
            assert(positionsCount == normalsCount);
            assert(tangentsCount == 0 || tangentsCount == positionsCount);
            assert(texCoord0sCount == 0 || texCoord0sCount == positionsCount);

            if (tangentsCount == 0)
                fprintf(
                    stderr,
                    "Missing tangents for '%s'. RT won't have normal "
                    "maps.\n",
                    mesh.name.c_str());

            const auto [indices, indexCount, usesShortIndices] = [&]
            {
                assert(primitive.indices > -1);
                const auto &accessor = gltfModel.accessors[primitive.indices];
                const auto &view = gltfModel.bufferViews[accessor.bufferView];
                const auto offset = asserted_cast<uint32_t>(
                    accessor.byteOffset + view.byteOffset);
                assert(
                    offset % sizeof(uint32_t) == 0 &&
                    "Shader binds buffers as uint");

                assert(
                    accessor.componentType ==
                        TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ||
                    accessor.componentType ==
                        TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);

                return std::make_tuple(
                    MeshBuffers::Buffer{
                        .index = asserted_cast<uint32_t>(view.buffer),
                        .offset =
                            offset / static_cast<uint32_t>(sizeof(uint32_t)),
                    },
                    asserted_cast<uint32_t>(accessor.count),
                    static_cast<uint32_t>(
                        accessor.componentType ==
                        TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT));
            }();

            // -1 is mapped to the default material
            assert(primitive.material > -2);
            const uint32_t material = primitive.material + 1;

            _meshBuffers.push_back(MeshBuffers{
                .indices = indices,
                .positions = positions,
                .normals = normals,
                .tangents = tangents,
                .texCoord0s = texCoord0s,
                .usesShortIndices = usesShortIndices,
            });
            _meshInfos.push_back(MeshInfo{
                .vertexCount = positionsCount,
                .indexCount = indexCount,
                .materialID = material,
            });

            model.subModels.push_back(Model::SubModel{
                .meshID = asserted_cast<uint32_t>(_meshBuffers.size() - 1),
                .materialID = material,
            });
        }
    }
    _meshBuffersBuffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = asserted_cast<uint32_t>(
                    _meshBuffers.size() * sizeof(MeshBuffers)),
                .usage = vk::BufferUsageFlagBits::
                             eAccelerationStructureBuildInputReadOnlyKHR |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .initialData = _meshBuffers.data(),
        .debugName = "MeshBuffersBuffer",
    });
}

void World::loadScenes(
    ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel)
{
    HashMap<Scene::Node *, size_t> lights{
        scopeAlloc, gltfModel.lights.size() * 2};
    // TODO: More complex nodes
    _nodes.reserve(gltfModel.nodes.size());

    for (const tinygltf::Node &gltfNode : gltfModel.nodes)
    {
        _nodes.emplace_back(_linearAlloc);
        Scene::Node &node = _nodes.back();

        node.children.reserve(gltfNode.children.size());
        for (const int child : gltfNode.children)
            node.children.push_back(&_nodes[child]);

        if (gltfNode.mesh > -1)
            node.modelID = gltfNode.mesh;
        if (gltfNode.camera > -1)
        {
            const auto &cam = gltfModel.cameras[gltfNode.camera];
            if (cam.type == "perspective")
                _cameras.insert_or_assign(
                    &node, CameraParameters{
                               .fov = static_cast<float>(cam.perspective.yfov),
                               .zN = static_cast<float>(cam.perspective.znear),
                               .zF = static_cast<float>(cam.perspective.zfar),
                           });
            else
                fprintf(
                    stderr, "Camera type '%s' is not supported\n",
                    cam.type.c_str());
        }
        if (gltfNode.extensions.contains("KHR_lights_punctual"))
        {
            // operator[] doesn't work for some reason
            const auto &ext = gltfNode.extensions.at("KHR_lights_punctual");
            const auto &obj = ext.Get<tinygltf::Value::Object>();

            const auto &light = obj.find("light")->second;
            assert(light.IsInt());

            lights.insert_or_assign(
                &node, asserted_cast<size_t>(light.GetNumberAsInt()));
        }
        if (gltfNode.matrix.size() == 16)
        {
            // Spec defines the matrix to be decomposeable to T * R * S
            const auto matrix = mat4{make_mat4(gltfNode.matrix.data())};
            vec3 skew;
            vec4 perspective;
            decompose(
                matrix, node.scale, node.rotation, node.translation, skew,
                perspective);
        }
        if (gltfNode.translation.size() == 3)
            node.translation = vec3{make_vec3(gltfNode.translation.data())};
        if (gltfNode.rotation.size() == 4)
            node.rotation = make_quat(gltfNode.rotation.data());
        if (gltfNode.scale.size() == 3)
            node.scale = vec3{make_vec3(gltfNode.scale.data())};
    }

    _scenes.reserve(gltfModel.scenes.size());
    for (const tinygltf::Scene &gltfScene : gltfModel.scenes)
    {
        _scenes.emplace_back(_linearAlloc);
        Scene &scene = _scenes.back();

        scene.nodes.reserve(gltfScene.nodes.size());
        for (const int node : gltfScene.nodes)
            scene.nodes.push_back(&_nodes[node]);
    }
    _currentScene = max(gltfModel.defaultScene, 0);

    // Traverse scenes and generate model instances for snappier rendering
    // Pre-alloc worst case memory for the stacks since it shouldn't be too much
    Array<Scene::Node *> nodeStack{scopeAlloc, _nodes.size()};
    Array<mat4> parentTransforms{scopeAlloc, _nodes.size()};
    wheels::HashSet<Scene::Node *> visited{_linearAlloc, _nodes.size() * 2};
    parentTransforms.push_back(mat4{1.f});
    for (auto &scene : _scenes)
    {
        bool directionalLightFound = false;

        visited.clear();
        nodeStack.clear();
        for (Scene::Node *node : scene.nodes)
            nodeStack.push_back(node);

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
                visited.insert(node);

                for (Scene::Node *child : node->children)
                    nodeStack.push_back(child);

                const mat4 modelToWorld =
                    parentTransforms.back() *
                    translate(mat4{1.f}, node->translation) *
                    mat4_cast(node->rotation) * scale(mat4{1.f}, node->scale);

                const auto normalToWorld = transpose(inverse(modelToWorld));
                if (node->modelID != 0xFFFFFFFF)
                {
                    scene.modelInstances.push_back(ModelInstance{
                        .id = asserted_cast<uint32_t>(
                            scene.modelInstances.size()),
                        .modelID = node->modelID,
                        .transforms = {
                            .modelToWorld = modelToWorld,
                            .normalToWorld = normalToWorld,
                        }});
                    scene.rtInstanceCount += asserted_cast<uint32_t>(
                        _models[node->modelID].subModels.size());
                }
                if (CameraParameters *params = _cameras.find(node);
                    params != nullptr)
                {
                    scene.camera = *params;
                    scene.camera.eye =
                        vec3{modelToWorld *vec4{0.f, 0.f, 0.f, 1.f}};
                    // TODO: Halfway from camera to scene bb end if inside
                    // bb / halfway of bb if outside of bb?
                    scene.camera.target =
                        vec3{modelToWorld *vec4{0.f, 0.f, -1.f, 1.f}};
                    scene.camera.up = mat3{modelToWorld} * vec3{0.f, 1.f, 0.f};
                }
                if (size_t const *light_i = lights.find(node);
                    light_i != nullptr)
                {
                    const auto &light = gltfModel.lights[*light_i];
                    if (light.type == "directional")
                    {
                        if (directionalLightFound)
                        {
                            fprintf(
                                stderr,
                                "Found second directional light for a "
                                "scene."
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

                        scene.lights.pointLights.data.emplace_back();
                        auto &sceneLight = scene.lights.pointLights.data.back();

                        sceneLight.radianceAndRadius = vec4{radiance, radius};
                        sceneLight.position =
                            modelToWorld *vec4{0.f, 0.f, 0.f, 1.f};
                    }
                    else if (light.type == "spot")
                    {
                        scene.lights.spotLights.data.emplace_back();
                        auto &sceneLight = scene.lights.spotLights.data.back();

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
                            modelToWorld *vec4{0.f, 0.f, 0.f, 1.f};
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

        // Scatter random lights in the scene
        // {
        //     const vec3 minBounds{-10.f, 0.5f, -5.f};
        //     const vec3 maxBounds{10.f, 7.f, 5.f};
        //     auto rando = [](float min, float max) -> float {
        //         return std::rand() / static_cast<float>(RAND_MAX) *
        //                    (max - min) +
        //                min;
        //     };
        //     for (auto i = 0; i < 64; ++i)
        //     {
        //         // rando W -> radiance
        //         auto radiance =
        //             vec3{rando(1.f, 5.f), rando(1.f, 5.f),
        //             rando(1.f, 5.f)} / (4.f * glm::pi<float>());
        //         const auto luminance =
        //             dot(radiance, vec3{0.2126, 0.7152, 0.0722});
        //         const auto minLuminance = 0.01f;
        //         const auto radius = sqrt(luminance / minLuminance);

        //         auto &data = scene.lights.pointLights.data;
        //         const auto li = data.count++;
        //         auto &sceneLight = data.lights[li];
        //         sceneLight.radianceAndRadius = vec4{radiance, radius};
        //         sceneLight.position = vec4{
        //             rando(minBounds.x, maxBounds.x),
        //             rando(minBounds.y, maxBounds.y),
        //             rando(minBounds.z, maxBounds.z), 1.f};
        //     }
        // }

        // Honor scene lighting
        if (!directionalLightFound && (!scene.lights.pointLights.data.empty() ||
                                       !scene.lights.spotLights.data.empty()))
        {
            scene.lights.directionalLight.parameters.irradiance = vec4{0.f};
        }
    }
}

void World::createBlases()
{
    assert(_meshBuffers.size() == _meshInfos.size());
    _blases.resize(_meshBuffers.size());
    for (size_t i = 0; i < _blases.size(); ++i)
    {
        const auto &buffers = _meshBuffers[i];
        const auto &info = _meshInfos[i];
        auto &blas = _blases[i];
        // Basics from RT Gems II chapter 16

        const auto positionsAddr =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = _geometryBuffers[buffers.positions.index].handle,
            });
        const auto positionsOffset =
            buffers.positions.offset * sizeof(uint32_t);

        const auto indicesAddr =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = _geometryBuffers[buffers.indices.index].handle,
            });
        const auto indicesOffset = buffers.indices.offset * sizeof(uint32_t);

        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = positionsAddr + positionsOffset,
            .vertexStride = 3 * sizeof(float),
            .maxVertex = info.vertexCount,
            .indexType = buffers.usesShortIndices == 1u
                             ? vk::IndexType::eUint16
                             : vk::IndexType::eUint32,
            .indexData = indicesAddr + indicesOffset,
        };

        const auto &material = _materials[info.materialID];
        const vk::GeometryFlagsKHR geomFlags =
            material.alphaMode == Material::AlphaMode::Opaque
                ? vk::GeometryFlagBitsKHR::eOpaque
                : vk::GeometryFlagsKHR{};
        const vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = triangles,
            .flags = geomFlags,
        };
        const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = info.indexCount / 3,
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
            .desc =
                BufferDescription{
                    .byteSize = sizeInfo.accelerationStructureSize,
                    .usage = vk::BufferUsageFlagBits::
                                 eAccelerationStructureStorageKHR |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
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
            .desc =
                BufferDescription{
                    .byteSize = sizeInfo.buildScratchSize,
                    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
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

void World::createTlases(ScopedScratch scopeAlloc)
{
    _tlases.resize(_scenes.size());
    for (size_t i = 0; i < _tlases.size(); ++i)
    {
        const auto &scene = _scenes[i];
        auto &tlas = _tlases[i];
        // Basics from RT Gems II chapter 16

        Array<vk::AccelerationStructureInstanceKHR> instances{
            scopeAlloc, scene.rtInstanceCount};
        Array<Pair<const Model &, vk::TransformMatrixKHR>> modelInstances{
            scopeAlloc, scene.modelInstances.size()};

        for (const auto &mi : scene.modelInstances)
        {
            const auto &model = _models[mi.modelID];
            modelInstances.emplace_back(
                model, convertTransform(mi.transforms.modelToWorld));
        }

        uint32_t rti = 0;
        for (const auto &[model, trfn] : modelInstances)
        {
            for (const auto &sm : model.subModels)
            {
                const auto &blas = _blases[sm.meshID];
                assert(blas.handle != vk::AccelerationStructureKHR{});

                instances.push_back(vk::AccelerationStructureInstanceKHR{
                    .transform = trfn,
                    .instanceCustomIndex = rti++,
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
            .desc =
                BufferDescription{
                    .byteSize = sizeof(instances[0]) * instances.size(),
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::
                                 eAccelerationStructureBuildInputReadOnlyKHR,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .initialData = instances.data(),
            .debugName = "InstancesBuffer",
        });

        // Need a barrier here if a shared command buffer is used so that
        // the copy happens before the build

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
            .desc =
                BufferDescription{
                    .byteSize = sizeInfo.accelerationStructureSize,
                    .usage = vk::BufferUsageFlagBits::
                                 eAccelerationStructureStorageKHR |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
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
            .desc =
                BufferDescription{
                    .byteSize = sizeInfo.buildScratchSize,
                    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .debugName = "ScratchBuffer",
        });

        buildInfo.scratchData =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = scratchBuffer.handle,
            });

        const auto cb = _device->beginGraphicsCommands();

        const auto *pRangeInfo = &rangeInfo;
        // TODO: Use a single cb for instance buffer copies and builds for
        // all
        //       tlases need a barrier after buffer copy and build!
        cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

        _device->endGraphicsCommands(cb);

        _device->destroy(scratchBuffer);
        _device->destroy(instancesBuffer);
    }
}

void World::createBuffers()
{
    _materialsBuffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = _materials.size() * sizeof(_materials[0]),
                .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
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
                for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
                    scene.modelInstanceTransformsBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .desc =
                                BufferDescription{
                                    .byteSize = bufferSize,
                                    .usage =
                                        vk::BufferUsageFlagBits::eStorageBuffer,
                                    .properties = vk::MemoryPropertyFlagBits::
                                                      eHostVisible |
                                                  vk::MemoryPropertyFlagBits::
                                                      eHostCoherent,
                                },
                            .createMapped = true,
                            .debugName = "InstanceTransforms",
                        }));
            }

            scene.rtInstancesBuffer = _device->createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize =
                            sizeof(Scene::RTInstance) * scene.rtInstanceCount,
                        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent,
                    },
                .createMapped = true,
                .debugName = "RTInstances",
            });

            {
                const vk::DeviceSize bufferSize =
                    sizeof(DirectionalLight::Parameters);
                for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
                    scene.lights.directionalLight.uniformBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .desc =
                                BufferDescription{
                                    .byteSize = bufferSize,
                                    .usage =
                                        vk::BufferUsageFlagBits::eUniformBuffer,
                                    .properties = vk::MemoryPropertyFlagBits::
                                                      eHostVisible |
                                                  vk::MemoryPropertyFlagBits::
                                                      eHostCoherent,
                                },
                            .createMapped = true,
                            .debugName = "DirectionalLightUniforms",
                        }));
            }

            {
                for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
                    scene.lights.pointLights.storageBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .desc =
                                BufferDescription{
                                    .byteSize = PointLights::sBufferByteSize,
                                    .usage =
                                        vk::BufferUsageFlagBits::eStorageBuffer,
                                    .properties = vk::MemoryPropertyFlagBits::
                                                      eHostVisible |
                                                  vk::MemoryPropertyFlagBits::
                                                      eHostCoherent,
                                },
                            .createMapped = true,
                            .debugName = "PointLightsBuffer",
                        }));
            }

            {
                for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
                    scene.lights.spotLights.storageBuffers.push_back(
                        _device->createBuffer(BufferCreateInfo{
                            .desc =
                                BufferDescription{
                                    .byteSize = SpotLights::sBufferByteSize,
                                    .usage =
                                        vk::BufferUsageFlagBits::eStorageBuffer,
                                    .properties = vk::MemoryPropertyFlagBits::
                                                      eHostVisible |
                                                  vk::MemoryPropertyFlagBits::
                                                      eHostCoherent,
                                },
                            .createMapped = true,
                            .debugName = "SpotLightsBuffer",
                        }));
            }
        }
    }

    {
        const vk::DeviceSize bufferSize = sizeof(mat4);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            _skyboxUniformBuffers.push_back(
                _device->createBuffer(BufferCreateInfo{
                    .desc =
                        BufferDescription{
                            .byteSize = bufferSize,
                            .usage = vk::BufferUsageFlagBits::eUniformBuffer,
                            .properties =
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent,
                        },
                    .createMapped = true,
                    .debugName = "SkyboxUniforms",
                }));
        }
    }
}

void World::createDescriptorSets(ScopedScratch scopeAlloc)
{
    if (_device == nullptr)
        throw std::runtime_error(
            "Tried to create World descriptor sets before loading glTF");

    //  We don't know the required capacity for this up front, let's not bleed
    //  reallocations in the linear scope allocator
    Array<vk::WriteDescriptorSet> dss{_generalAlloc};

    // Materials layout and descriptors set
    // Define outside the helper scope to keep alive until
    // updateDescriptorSets
    const vk::DescriptorBufferInfo materialDatasInfo{
        .buffer = _materialsBuffer.handle,
        .range = VK_WHOLE_SIZE,
    };
    Array<vk::DescriptorImageInfo> materialSamplerInfos{
        scopeAlloc, _samplers.size()};
    // Use capacity instead of size so that this allocates descriptors for
    // textures that are loaded later
    Array<vk::DescriptorImageInfo> materialImageInfos{
        scopeAlloc, _texture2Ds.capacity()};
    {
        for (const auto &s : _samplers)
            materialSamplerInfos.push_back(
                vk::DescriptorImageInfo{.sampler = s});
        const auto samplerInfoCount =
            asserted_cast<uint32_t>(materialSamplerInfos.size());
        _dsLayouts.materialSamplerCount = samplerInfoCount;

        if (_deferredLoadingContext.has_value())
        {
            // Fill missing textures with the default info so potential reads
            // are still to valid descriptors
            assert(_texture2Ds.size() == 1);
            const vk::DescriptorImageInfo defaultInfo =
                _texture2Ds[0].imageInfo();
            for (size_t i = 0; i < materialImageInfos.capacity(); ++i)
                materialImageInfos.push_back(defaultInfo);
        }
        else
        {
            for (const auto &tex : _texture2Ds)
                materialImageInfos.push_back(tex.imageInfo());
        }

        const auto imageInfoCount =
            asserted_cast<uint32_t>(materialImageInfos.size());

        const StaticArray layoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = samplerInfoCount,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1 + samplerInfoCount,
                .descriptorType = vk::DescriptorType::eSampledImage,
                .descriptorCount = imageInfoCount,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
        };
        const StaticArray layoutFlags{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
                // Texture bindings for deferred loads are updated before
                // frame cb submission, for textures that aren't accessed any
                // frame in flight
                vk::DescriptorBindingFlagBits::ePartiallyBound |
                vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending},
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

        _materialTexturesDS = _descriptorAllocator.allocate(
            _dsLayouts.materialTextures, imageInfoCount);

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

        const uint32_t textureArrayBinding =
            1 + asserted_cast<uint32_t>(materialSamplerInfos.size());
        if (_deferredLoadingContext.has_value())
            _deferredLoadingContext->textureArrayBinding = textureArrayBinding;

        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _materialTexturesDS,
            .dstBinding = textureArrayBinding,
            .dstArrayElement = 0,
            .descriptorCount =
                asserted_cast<uint32_t>(materialImageInfos.size()),
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = materialImageInfos.data(),
        });
    }

    // Geometry layouts and descriptor set
    // Define outside the helper scope to keep alive until
    // updateDescriptorSets
    Array<vk::DescriptorBufferInfo> bufferInfos{
        scopeAlloc, _geometryBuffers.size()};
    {
        for (const auto &b : _geometryBuffers)
            bufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = b.handle,
                .range = VK_WHOLE_SIZE,
            });
        const auto bufferCount = asserted_cast<uint32_t>(bufferInfos.size());

        bufferInfos.push_back(vk::DescriptorBufferInfo{
            .buffer = _meshBuffersBuffer.handle,
            .range = VK_WHOLE_SIZE,
        });

        const StaticArray layoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex |
                              vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = bufferCount,
                .stageFlags = vk::ShaderStageFlagBits::eVertex |
                              vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
        };
        const StaticArray descriptorFlags = {
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount},
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
                    .bindingCount =
                        asserted_cast<uint32_t>(descriptorFlags.size()),
                    .pBindingFlags = descriptorFlags.data(),
                }};
        _dsLayouts.geometry = _device->logical().createDescriptorSetLayout(
            layoutChain.get<vk::DescriptorSetLayoutCreateInfo>());

        _geometryDS =
            _descriptorAllocator.allocate(_dsLayouts.geometry, bufferCount);

        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _geometryDS,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &bufferInfos.back(),
        });
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _geometryDS,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = asserted_cast<uint32_t>(bufferInfos.size() - 1),
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = bufferInfos.data(),
        });
    }

    // RT layout
    {
        const StaticArray layoutBindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
        };
        const vk::DescriptorSetLayoutCreateInfo createInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        };
        _dsLayouts.rayTracing =
            _device->logical().createDescriptorSetLayout(createInfo);
    }

    // Model instances layout
    {
        const vk::DescriptorSetLayoutBinding layoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex |
                          vk::ShaderStageFlagBits::eRaygenKHR |
                          vk::ShaderStageFlagBits::eAnyHitKHR,
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
        const StaticArray layoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute |
                              vk::ShaderStageFlagBits::eRaygenKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute |
                              vk::ShaderStageFlagBits::eRaygenKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment |
                              vk::ShaderStageFlagBits::eCompute |
                              vk::ShaderStageFlagBits::eRaygenKHR,
            },
        };
        _dsLayouts.lights = _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });
    }

    // Scene descriptor sets
    // Define outside the helper scope to keep alive until
    // updateDescriptorSets
    Array<vk::DescriptorBufferInfo> modelInstanceInfos{scopeAlloc};
    Array<vk::DescriptorBufferInfo> rtInstancesInfos{
        scopeAlloc, _scenes.size()};
    Array<vk::DescriptorBufferInfo> lightInfos{scopeAlloc};
    Array<vk::StructureChain<
        vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR>>
        asDSChains{scopeAlloc};
    // TODO: Reserve required memory upfront
    {
        for (auto &scene : _scenes)
        {
            {
                Array<vk::DescriptorSetLayout> layouts{
                    scopeAlloc, MAX_FRAMES_IN_FLIGHT};
                layouts.resize(MAX_FRAMES_IN_FLIGHT, _dsLayouts.modelInstances);
                scene.modelInstancesDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
                _descriptorAllocator.allocate(
                    layouts, Span{
                                 scene.modelInstancesDescriptorSets.data(),
                                 scene.modelInstancesDescriptorSets.size()});

                modelInstanceInfos.reserve(
                    modelInstanceInfos.size() +
                    scene.modelInstanceTransformsBuffers.size());
                const auto startIndex =
                    asserted_cast<uint32_t>(modelInstanceInfos.size());
                for (auto &buffer : scene.modelInstanceTransformsBuffers)
                    modelInstanceInfos.push_back(vk::DescriptorBufferInfo{
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
                Array<vk::DescriptorSetLayout> layouts{
                    scopeAlloc, MAX_FRAMES_IN_FLIGHT};
                layouts.resize(MAX_FRAMES_IN_FLIGHT, _dsLayouts.lights);

                auto &lights = scene.lights;
                lights.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
                _descriptorAllocator.allocate(
                    layouts, Span{
                                 lights.descriptorSets.data(),
                                 lights.descriptorSets.size()});

                StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT>
                    dirLightInfos;
                dirLightInfos.resize(
                    lights.directionalLight.uniformBuffers.size());
                lights.directionalLight.bufferInfos(dirLightInfos);

                StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT>
                    pointLightInfos;
                pointLightInfos.resize(
                    lights.pointLights.storageBuffers.size());
                lights.pointLights.bufferInfos(pointLightInfos);

                StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT>
                    spotLightInfos;
                spotLightInfos.resize(lights.spotLights.storageBuffers.size());
                lights.spotLights.bufferInfos(spotLightInfos);

                const auto dirLightStart = lightInfos.size();
                const auto pointLightStart =
                    dirLightStart + dirLightInfos.size();
                const auto spotLightStart =
                    pointLightStart + pointLightInfos.size();

                lightInfos.reserve(
                    lightInfos.size() + dirLightInfos.size() +
                    pointLightInfos.size() + spotLightInfos.size());
                // WHEELSTODO: Array::extend(Array const&)
                for (const auto &info : dirLightInfos)
                    lightInfos.push_back(info);
                for (const auto &info : pointLightInfos)
                    lightInfos.push_back(info);
                for (const auto &info : spotLightInfos)
                    lightInfos.push_back(info);

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
                scene.rtDescriptorSet =
                    _descriptorAllocator.allocate(_dsLayouts.rayTracing);

                asDSChains.emplace_back(
                    vk::WriteDescriptorSet{
                        .dstSet = scene.rtDescriptorSet,
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

                rtInstancesInfos.push_back(vk::DescriptorBufferInfo{
                    .buffer = scene.rtInstancesBuffer.handle,
                    .range = VK_WHOLE_SIZE});
                dss.push_back(vk::WriteDescriptorSet{
                    .dstSet = scene.rtDescriptorSet,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &rtInstancesInfos.back(),
                });
            }
        }
    }

    // Skybox layout and descriptor sets
    StaticArray<vk::DescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT>
        skyboxBufferInfos;
    vk::DescriptorImageInfo skyboxImageInfo;
    {
        const StaticArray skyboxLayoutBindings{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        _dsLayouts.skybox = _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount =
                    asserted_cast<uint32_t>(skyboxLayoutBindings.size()),
                .pBindings = skyboxLayoutBindings.data(),
            });

        const vk::DescriptorSetLayoutBinding skyboxOnlyLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
        };
        _dsLayouts.skyboxOnly = _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = 1,
                .pBindings = &skyboxOnlyLayoutBinding,
            });

        StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT>
            skyboxLayouts;
        skyboxLayouts.resize(MAX_FRAMES_IN_FLIGHT, _dsLayouts.skybox);
        _skyboxDSs.resize(MAX_FRAMES_IN_FLIGHT);
        _descriptorAllocator.allocate(skyboxLayouts, _skyboxDSs);

        _skyboxOnlyDS = _descriptorAllocator.allocate(_dsLayouts.skyboxOnly);

        for (auto &buffer : _skyboxUniformBuffers)
            skyboxBufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = buffer.handle,
                .offset = 0,
                .range = sizeof(mat4),
            });
        skyboxImageInfo = _skyboxTexture.imageInfo();

        dss.reserve(dss.size() + _skyboxDSs.size() * 2 + 1);
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
        dss.push_back(vk::WriteDescriptorSet{
            .dstSet = _skyboxOnlyDS,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &skyboxImageInfo,
        });
    }

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(dss.size()), dss.data(), 0, nullptr);
}

World::DeferredLoadingContext::DeferredLoadingContext(
    Allocator &alloc, Device *device, const tinygltf::Model &gltfModel)
: device{device}
, gltfModel{gltfModel}
, materials{alloc, gltfModel.materials.size()}
{
    assert(device != nullptr);
    for (uint32_t i = 0; i < stagingBuffers.capacity(); ++i)
        stagingBuffers.push_back(createTextureStaging(device));
}

World::DeferredLoadingContext::~DeferredLoadingContext()
{
    if (device != nullptr)
        for (const Buffer &buffer : stagingBuffers)
            device->destroy(buffer);
}
