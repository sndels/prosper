#include "World.hpp"

#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <cstdlib>
#include <imgui.h>
#include <iostream>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/pair.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/VkUtils.hpp"
#include "../utils/Timer.hpp"
#include "../utils/Utils.hpp"

using namespace glm;
using namespace wheels;

#ifdef _WIN32
// Windows' header doesn't include these
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_MIRRORED_REPEAT 0x8370
#endif // _WIN32 or _WIN64

namespace
{
constexpr uint32_t sMaterialDatasReflectionSet = 0;
constexpr uint32_t sMaterialTexturesReflectionSet = 1;
constexpr uint32_t sGeometryReflectionSet = 0;
constexpr uint32_t sInstanceTrfnsReflectionSet = 0;
constexpr uint32_t sLightsReflectionSet = 0;
constexpr uint32_t sSkyboxReflectionSet = 0;

constexpr size_t sWorldMemSize = megabytes(16);

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
    const vk::DeviceSize stagingSize = static_cast<size_t>(4096) *
                                       static_cast<size_t>(4096) *
                                       sizeof(uint32_t);
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

void loadingWorker(
    const std::filesystem::path *sceneDir, World::DeferredLoadingContext *ctx)
{
    assert(sceneDir != nullptr);
    assert(ctx != nullptr);
    assert(ctx->device != nullptr);
    assert(ctx->device->transferQueue().has_value());
    assert(ctx->device->graphicsQueue() != *ctx->device->transferQueue());

    // Enough for 4K textures, it seems
    LinearAllocator scratchBacking{megabytes(256)};
    ScopedScratch scopeAlloc{scratchBacking};

    while (!ctx->interruptLoading)
    {
        if (ctx->workerLoadedImageCount == ctx->gltfModel.images.size())
            break;

        assert(ctx->gltfModel.images.size() > ctx->workerLoadedImageCount);
        const tinygltf::Image &image =
            ctx->gltfModel.images[ctx->workerLoadedImageCount];
        if (image.uri.empty())
            throw std::runtime_error("Embedded glTF textures aren't supported. "
                                     "Scene should be glTF + "
                                     "bin + textures.");

        ctx->cb.reset();
        ctx->cb.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });

        Texture2D tex{
            scopeAlloc.child_scope(),
            ctx->device,
            *sceneDir / image.uri,
            ctx->cb,
            ctx->stagingBuffers[0],
            true,
            true};

        const QueueFamilies &families = ctx->device->queueFamilies();
        assert(families.graphicsFamily.has_value());
        assert(families.transferFamily.has_value());

        if (*families.graphicsFamily != *families.transferFamily)
        {
            const vk::ImageMemoryBarrier2 releaseBarrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eNone,
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = *families.transferFamily,
                .dstQueueFamilyIndex = *families.graphicsFamily,
                .image = tex.nativeHandle(),
                .subresourceRange =
                    vk::ImageSubresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            ctx->cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &releaseBarrier,
            });
        }

        ctx->cb.end();

        const vk::Queue transferQueue = *ctx->device->transferQueue();
        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &ctx->cb,
        };
        checkSuccess(
            transferQueue.submit(1, &submitInfo, vk::Fence{}),
            "submitTextureUpload");
        // We could have multiple uploads in flight, but let's be simple for now
        transferQueue.waitIdle();

        ctx->workerLoadedImageCount++;

        const uint32_t previousHighWatermark = ctx->allocationHighWatermark;
        if (scratchBacking.allocated_byte_count_high_watermark() >
            previousHighWatermark)
        {
            ctx->allocationHighWatermark = asserted_cast<uint32_t>(
                scratchBacking.allocated_byte_count_high_watermark());
        }

        {
            std::unique_lock lock{ctx->loadedTextureMutex};

            if (ctx->loadedTexture.has_value())
                ctx->loadedTextureTaken.wait(lock);
            assert(!ctx->loadedTexture.has_value());

            ctx->loadedTexture.emplace(WHEELS_MOV(tex));

            lock.unlock();
        }
    }
}

struct Node
{
    Array<uint32_t> children;
    glm::vec3 translation{0.f};
    glm::quat rotation{1.f, 0.f, 0.f, 0.f};
    glm::vec3 scale{1.f};
    Optional<uint32_t> modelID;
    Optional<uint32_t> camera;
    Optional<uint32_t> light;

    Node(wheels::Allocator &alloc)
    : children{alloc}
    {
    }
};

} // namespace

World::World(
    Allocator &generalAlloc, ScopedScratch scopeAlloc, Device *device,
    const std::filesystem::path &scene, bool deferredLoading)
: _generalAlloc{generalAlloc}
, _linearAlloc{sWorldMemSize}
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

    // Deferred loading is used for textures only
    deferredLoading = deferredLoading && !gltfModel.textures.empty();

    if (deferredLoading)
        _deferredLoadingContext.emplace(
            _generalAlloc, _device, &_sceneDir, gltfModel);

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

    reflectBindings(scopeAlloc.child_scope());
    createDescriptorSets(scopeAlloc.child_scope());
}

World::~World()
{
    _device->logical().destroy(_dsLayouts.lights);
    _device->logical().destroy(_dsLayouts.skybox);
    _device->logical().destroy(_dsLayouts.rayTracing);
    _device->logical().destroy(_dsLayouts.modelInstances);
    _device->logical().destroy(_dsLayouts.geometry);
    _device->logical().destroy(_dsLayouts.materialTextures);
    _device->logical().destroy(_dsLayouts.materialDatas);

    _device->destroy(_skyboxVertexBuffer);
    for (auto &buffer : _materialsBuffers)
        _device->destroy(buffer);

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
        _device->destroy(scene.rtInstancesBuffer);
    for (auto &buffer : _geometryBuffers)
        _device->destroy(buffer);
    _device->destroy(_meshBuffersBuffer);
    for (auto &sampler : _samplers)
        _device->logical().destroy(sampler);

    _device->destroy(_scratchBuffer);
    _device->destroy(_tlasInstancesBuffer);
}

void World::startFrame() const
{
    _modelInstanceTransformsRing->startFrame();
    _lightDataRing->startFrame();
    _tlasInstancesUploadRing->startFrame();
}

void World::uploadMaterialDatas(uint32_t nextFrame)
{
    if (_deferredLoadingContext.has_value())
    {
        if (_materialsGenerations[nextFrame] !=
            _deferredLoadingContext->materialsGeneration)
        {
            Material *mapped = reinterpret_cast<Material *>(
                _materialsBuffers[nextFrame].mapped);
            memcpy(
                mapped, _materials.data(),
                _materials.size() * sizeof(_materials[0]));

            _materialsGenerations[nextFrame] =
                _deferredLoadingContext->materialsGeneration;
        }
    }
}

void World::handleDeferredLoading(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame,
    Profiler &profiler)
{
    if (!_deferredLoadingContext.has_value())
        return;

    _deferredLoadingAllocationHighWatermark = std::max(
        _deferredLoadingAllocationHighWatermark,
        _deferredLoadingContext->allocationHighWatermark.load());

    if (_deferredLoadingContext->loadedMaterialCount ==
        _deferredLoadingContext->gltfModel.materials.size())
    {
        // Don't clean up until all in flight uploads are finished
        if (_deferredLoadingContext->framesSinceFinish++ > MAX_FRAMES_IN_FLIGHT)
        {
            printf(
                "Material streaming took %.2fs\n",
                _deferredLoadingContext->timer.getSeconds());

            _deferredLoadingContext.reset();
        }
        return;
    }

    // No gpu as timestamps are flaky for this work
    const auto _s = profiler.createCpuScope("DeferredLoading");

    if (_deferredLoadingContext->loadedImageCount == 0)
        _deferredLoadingContext->timer.reset();

    bool newTextureAvailable = false;
    if (_deferredLoadingContext->worker.has_value())
        newTextureAvailable = pollTextureWorker(cb);
    else
    {
        loadTextureSingleThreaded(scopeAlloc.child_scope(), cb, nextFrame);
        newTextureAvailable = true;
    }

    if (newTextureAvailable)
        updateDescriptorsWithNewTexture();
}

void World::drawDeferredLoadingUi() const
{
    if (_deferredLoadingContext.has_value())
    {
        ImGui::SetNextWindowPos(ImVec2{400, 50}, ImGuiCond_Appearing);
        ImGui::Begin(
            "DeferredLoadingProgress", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text(
            "Images loaded: %u/%u", _deferredLoadingContext->loadedImageCount,
            asserted_cast<uint32_t>(
                _deferredLoadingContext->gltfModel.images.size()));
        ImGui::End();
    }
}

Scene &World::currentScene() { return _scenes[_currentScene]; }

const Scene &World::currentScene() const { return _scenes[_currentScene]; }

void World::updateScene(ScopedScratch scopeAlloc, Profiler *profiler)
{
    assert(profiler != nullptr);

    auto _s = profiler->createCpuScope("World::updateScene");

    Scene &scene = currentScene();

    Array<uint32_t> nodeStack{scopeAlloc, scene.nodes.size()};
    Array<mat4> parentTransforms{scopeAlloc, scene.nodes.size()};
    wheels::HashSet<uint32_t> visited{scopeAlloc, scene.nodes.size()};
    for (uint32_t rootIndex : scene.rootNodes)
    {
        nodeStack.clear();
        parentTransforms.clear();
        visited.clear();

        nodeStack.push_back(rootIndex);
        parentTransforms.push_back(mat4{1.f});
        while (!nodeStack.empty())
        {
            const uint32_t nodeIndex = nodeStack.back();
            if (visited.find(nodeIndex) != visited.end())
            {
                nodeStack.pop_back();
                parentTransforms.pop_back();
            }
            else
            {
                visited.insert(nodeIndex);
                Scene::Node &node = scene.nodes[nodeIndex];

                const uint32_t first_child = node.firstChild;
                const uint32_t last_child = node.lastChild;
                for (uint32_t child = first_child; child <= last_child; ++child)
                    nodeStack.push_back(child);

                // TODO:
                // Skip identity SRT components?
                const mat4 modelToWorld4x4 =
                    parentTransforms.back() *
                    translate(mat4{1.f}, node.translation) *
                    mat4_cast(node.rotation) * scale(mat4{1.f}, node.scale);

                const mat3x4 modelToWorld = transpose(modelToWorld4x4);
                // No transpose as mat4->mat3x4 effectively does it
                const mat3x4 normalToWorld = inverse(modelToWorld4x4);

                if (node.modelInstance.has_value())
                    scene.modelInstances[*node.modelInstance].transforms =
                        ModelInstance::Transforms{
                            .modelToWorld = modelToWorld,
                            .normalToWorld = normalToWorld,
                        };

                if (node.camera.has_value() && *node.camera == _currentCamera)
                {
                    scene.camera.eye =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f}};
                    // TODO: Halfway from camera to scene bb end if inside
                    // bb / halfway of bb if outside of bb?
                    scene.camera.target =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, -1.f, 1.f}};
                    scene.camera.up =
                        mat3{modelToWorld4x4} * vec3{0.f, 1.f, 0.f};
                }

                if (node.directionalLight)
                {
                    auto &parameters = scene.lights.directionalLight.parameters;
                    parameters.direction =
                        vec4{mat3{modelToWorld4x4} * vec3{0.f, 0.f, -1.f}, 0.f};
                }

                if (node.pointLight.has_value())
                {
                    PointLight &sceneLight =
                        scene.lights.pointLights.data[*node.pointLight];

                    sceneLight.position =
                        modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f};
                }

                if (node.spotLight.has_value())
                {
                    SpotLight &sceneLight =
                        scene.lights.spotLights.data[*node.spotLight];

                    const vec3 position =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f}};
                    sceneLight.positionAndAngleOffset.x = position.x;
                    sceneLight.positionAndAngleOffset.y = position.y;
                    sceneLight.positionAndAngleOffset.z = position.z;

                    sceneLight.direction =
                        vec4{mat3{modelToWorld4x4} * vec3{0.f, 0.f, -1.f}, 0.f};
                }
                parentTransforms.emplace_back(modelToWorld4x4);
            }
        }
    }
}

void World::updateBuffers(ScopedScratch scopeAlloc)
{
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

        _modelInstanceTransformsByteOffset =
            _modelInstanceTransformsRing->write_elements(transforms);

        memcpy(
            scene.rtInstancesBuffer.mapped, rtInstances.data(),
            sizeof(Scene::RTInstance) * rtInstances.size());
    }

    updateTlasInstances(scopeAlloc.child_scope(), scene);

    _directionalLightByteOffset =
        scene.lights.directionalLight.write(*_lightDataRing);
    _pointLightByteOffset = scene.lights.pointLights.write(*_lightDataRing);
    _spotLightByteOffset = scene.lights.spotLights.write(*_lightDataRing);
}

void World::buildCurrentTlas(vk::CommandBuffer cb)
{
    const auto &scene = _scenes[_currentScene];
    auto &tlas = _tlases[_currentScene];

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
    vk::AccelerationStructureGeometryKHR geometry;
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo;
    createTlasBuildInfos(scene, rangeInfo, geometry, buildInfo, sizeInfo);

    buildInfo.dstAccelerationStructure = tlas.handle;

    reserveScratch(sizeInfo.buildScratchSize);

    buildInfo.scratchData =
        _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
            .buffer = _scratchBuffer.handle,
        });

    const vk::BufferCopy copyRegion{
        .srcOffset = _tlasInstancesUploadOffset,
        .dstOffset = 0,
        .size = _tlasInstancesBuffer.byteSize,
    };
    cb.copyBuffer(
        _tlasInstancesUploadRing->buffer(), _tlasInstancesBuffer.handle, 1,
        &copyRegion);

    const vk::AccelerationStructureBuildRangeInfoKHR *pRangeInfo = &rangeInfo;
    cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

    // First use needs to have a memory barrier from AS build into the usage
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
            scopeAlloc.child_scope(), _device, resPath("texture/empty.png"), cb,
            stagingBuffer, false);
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
            if (image.uri.empty())
                throw std::runtime_error("Embedded glTF textures aren't "
                                         "supported. Scene should be glTF + "
                                         "bin + textures.");

            const vk::CommandBuffer cb = _device->beginGraphicsCommands();

            _texture2Ds.emplace_back(
                scopeAlloc.child_scope(), _device, _sceneDir / image.uri, cb,
                stagingBuffer, true);

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
    // Parse raw nodes first so conversion to internal format happens only once
    // for potential instances
    Array<Node> nodes{scopeAlloc, gltfModel.nodes.size()};
    for (const tinygltf::Node &gltfNode : gltfModel.nodes)
    {
        nodes.emplace_back(_linearAlloc);
        Node &node = nodes.back();

        node.children.reserve(gltfNode.children.size());
        for (const int child : gltfNode.children)
            node.children.push_back(asserted_cast<uint32_t>(child));

        if (gltfNode.mesh > -1)
            node.modelID = gltfNode.mesh;
        if (gltfNode.camera > -1)
        {
            const uint32_t cameraIndex = gltfNode.camera;
            const auto &cam = gltfModel.cameras[cameraIndex];
            if (cam.type == "perspective")
            {
                if (_cameras.size() <= cameraIndex)
                    _cameras.resize(cameraIndex + 1);

                _cameras[cameraIndex] = CameraParameters{
                    .fov = static_cast<float>(cam.perspective.yfov),
                    .zN = static_cast<float>(cam.perspective.znear),
                    .zF = static_cast<float>(cam.perspective.zfar),
                };

                node.camera = cameraIndex;
            }
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

            node.light = asserted_cast<uint32_t>(light.GetNumberAsInt());
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

    _currentScene = max(gltfModel.defaultScene, 0);

    struct NodePair
    {
        uint32_t tmpNode{0xFFFFFFFF};
        uint32_t sceneNode{0xFFFFFFFF};
    };
    Array<NodePair> nodeStack{scopeAlloc, nodes.size()};
    // Traverse scene trees and generate actual scene datas
    // Traverse scenes and generate model instances for snappier rendering
    // Pre-alloc worst case memory for the stacks since it shouldn't be too much
    _scenes.reserve(gltfModel.scenes.size());
    for (const tinygltf::Scene &gltfScene : gltfModel.scenes)
    {
        _scenes.emplace_back(_linearAlloc);
        Scene &scene = _scenes.back();

        bool directionalLightFound = false;

        for (const int nodeIndex : gltfScene.nodes)
        {
            // Our node indices don't match gltf's anymore, push index of the
            // new node into roots
            scene.rootNodes.push_back(
                asserted_cast<uint32_t>(scene.nodes.size()));
            scene.nodes.emplace_back();

            // Start adding nodes from the new root
            nodeStack.clear();
            nodeStack.emplace_back(
                asserted_cast<uint32_t>(nodeIndex),
                asserted_cast<uint32_t>(scene.nodes.size() - 1));
            while (!nodeStack.empty())
            {
                const NodePair indices = nodeStack.pop_back();
                Node &tmpNode = nodes[indices.tmpNode];

                // Push children to the back of nodes before getting the current
                // node's reference to avoid it invalidating
                const uint32_t childCount =
                    asserted_cast<uint32_t>(tmpNode.children.size());
                const uint32_t firstChild =
                    asserted_cast<uint32_t>(scene.nodes.size());
                // If no children, firstChild <= lastChild false as intended.
                const uint32_t lastChild = firstChild + childCount - 1;
                scene.nodes.resize(
                    scene.nodes.size() + asserted_cast<size_t>(childCount));

                Scene::Node &sceneNode = scene.nodes[indices.sceneNode];
                sceneNode.firstChild = firstChild;
                sceneNode.lastChild = lastChild;

                for (uint32_t i = 0; i < childCount; ++i)
                {
                    const uint32_t childIndex = sceneNode.firstChild + i;
                    scene.nodes[childIndex].parent = indices.sceneNode;
                    nodeStack.emplace_back(
                        asserted_cast<uint32_t>(tmpNode.children[i]),
                        asserted_cast<uint32_t>(childIndex));
                }

                sceneNode.translation = tmpNode.translation;
                sceneNode.rotation = tmpNode.rotation;
                sceneNode.scale = tmpNode.scale;
                sceneNode.modelID = tmpNode.modelID;
                sceneNode.camera = tmpNode.camera;

                if (sceneNode.modelID.has_value())
                {
                    sceneNode.modelInstance =
                        asserted_cast<uint32_t>(scene.modelInstances.size());
                    // TODO:
                    // Why is id needed here? It's just the index in the array
                    scene.modelInstances.push_back(ModelInstance{
                        .id = *sceneNode.modelInstance,
                        .modelID = *sceneNode.modelID,
                    });
                    scene.rtInstanceCount += asserted_cast<uint32_t>(
                        _models[*sceneNode.modelID].subModels.size());
                }

                if (tmpNode.light.has_value())
                {
                    const tinygltf::Light &light =
                        gltfModel.lights[*tmpNode.light];
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

                        sceneNode.directionalLight = true;
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

                        sceneNode.pointLight = asserted_cast<uint32_t>(
                            scene.lights.pointLights.data.size());
                        scene.lights.pointLights.data.emplace_back();
                        auto &sceneLight = scene.lights.pointLights.data.back();

                        sceneLight.radianceAndRadius = vec4{radiance, radius};
                    }
                    else if (light.type == "spot")
                    {
                        sceneNode.spotLight = asserted_cast<uint32_t>(
                            scene.lights.spotLights.data.size());
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

                        sceneLight.positionAndAngleOffset.w = angleOffset;
                    }
                    else
                    {
                        fprintf(
                            stderr, "Unknown light type '%s'\n",
                            light.type.c_str());
                    }
                }
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

        reserveScratch(sizeInfo.buildScratchSize);

        buildInfo.scratchData =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = _scratchBuffer.handle,
            });

        const auto cb = _device->beginGraphicsCommands();

        const auto *pRangeInfo = &rangeInfo;
        // TODO: Build multiple blas at a time/with the same cb
        cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

        _device->endGraphicsCommands(cb);
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

        if (_tlasInstancesUploadRing != nullptr)
            // Not really a start of a frame but ring usage is validated
            _tlasInstancesUploadRing->startFrame();
        updateTlasInstances(scopeAlloc.child_scope(), scene);

        vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
        vk::AccelerationStructureGeometryKHR geometry;
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
        vk::AccelerationStructureBuildSizesInfoKHR sizeInfo;
        createTlasBuildInfos(scene, rangeInfo, geometry, buildInfo, sizeInfo);

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
    }

    // Reset ring to avoid confusion when comparing to other rings after loading
    _tlasInstancesUploadRing->reset();
}

void World::reserveScratch(vk::DeviceSize byteSize)
{
    if (_scratchBuffer.byteSize < byteSize)
    {
        _device->destroy(_scratchBuffer);
        _scratchBuffer = _device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = byteSize,
                    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .debugName = "ScratchBuffer",
        });
    }
}

void World::reserveTlasInstances(
    wheels::Span<const vk::AccelerationStructureInstanceKHR> instances)
{
    const vk::DeviceSize byteSize = sizeof(instances[0]) * instances.size();
    if (_tlasInstancesBuffer.byteSize < byteSize)
    {
        _device->destroy(_tlasInstancesBuffer);
        _tlasInstancesUploadRing.reset();

        _tlasInstancesBuffer = _device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = byteSize,
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::
                                 eAccelerationStructureBuildInputReadOnlyKHR,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .debugName = "InstancesBuffer",
        });

        const uint32_t ringByteSize = asserted_cast<uint32_t>(
            (byteSize + RingBuffer::sAlignment) * MAX_FRAMES_IN_FLIGHT);
        _tlasInstancesUploadRing = std::make_unique<RingBuffer>(
            _device, vk::BufferUsageFlagBits::eTransferSrc, ringByteSize,
            "InstancesUploadBuffer");
        _tlasInstancesUploadRing->startFrame();
    }
}

void World::updateTlasInstances(
    wheels::ScopedScratch scopeAlloc, const Scene &scene)
{
    // TODO:
    // Is it faster to poke instances directly into a mapped buffer instead
    // of collecting first and then passing them in one blob as initial
    // data?
    Array<vk::AccelerationStructureInstanceKHR> instances{
        scopeAlloc, scene.rtInstanceCount};
    uint32_t rti = 0;
    for (const auto &mi : scene.modelInstances)
    {
        const auto &model = _models[mi.modelID];

        // This has to be mat3x4 because we assume the transform already has
        // the same memory layout as vk::TransformationMatrixKHR
        const mat3x4 *trfn = &mi.transforms.modelToWorld;
        const vk::TransformMatrixKHR *trfn_cast =
            reinterpret_cast<const vk::TransformMatrixKHR *>(trfn);

        for (const auto &sm : model.subModels)
        {
            const auto &blas = _blases[sm.meshID];
            assert(blas.handle != vk::AccelerationStructureKHR{});

            instances.push_back(vk::AccelerationStructureInstanceKHR{
                .transform = *trfn_cast,
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
    assert(instances.size() == scene.rtInstanceCount);

    reserveTlasInstances(instances);

    _tlasInstancesUploadOffset =
        _tlasInstancesUploadRing->write_elements(instances);
}

void World::createTlasBuildInfos(
    const Scene &scene,
    vk::AccelerationStructureBuildRangeInfoKHR &rangeInfoOut,
    vk::AccelerationStructureGeometryKHR &geometryOut,
    vk::AccelerationStructureBuildGeometryInfoKHR &buildInfoOut,
    vk::AccelerationStructureBuildSizesInfoKHR &sizeInfoOut)
{
    rangeInfoOut = vk::AccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = scene.rtInstanceCount,
        .primitiveOffset = 0,
    };

    geometryOut = vk::AccelerationStructureGeometryKHR{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry =
            vk::AccelerationStructureGeometryInstancesDataKHR{
                .data = _device->logical().getBufferAddress(
                    vk::BufferDeviceAddressInfo{
                        .buffer = _tlasInstancesBuffer.handle,
                    }),
            },
    };

    buildInfoOut = vk::AccelerationStructureBuildGeometryInfoKHR{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geometryOut,
    };

    sizeInfoOut = _device->logical().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfoOut,
        {rangeInfoOut.primitiveCount});
}

void World::createBuffers()
{
    for (size_t i = 0; i < _materialsBuffers.capacity(); ++i)
        _materialsBuffers.push_back(_device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = _materials.size() * sizeof(_materials[0]),
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .initialData = _materials.data(),
            .createMapped = true,
            .debugName = "MaterialsBuffer",
        }));

    {
        size_t maxModelInstanceTransforms = 0;
        for (auto &scene : _scenes)
        {
            maxModelInstanceTransforms = std::max(
                maxModelInstanceTransforms, scene.modelInstances.size());

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
        }

        const uint32_t bufferSize = asserted_cast<uint32_t>(
            (maxModelInstanceTransforms * sizeof(ModelInstance::Transforms) +
             static_cast<size_t>(RingBuffer::sAlignment)) *
            MAX_FRAMES_IN_FLIGHT);
        _modelInstanceTransformsRing = std::make_unique<RingBuffer>(
            _device, vk::BufferUsageFlagBits::eStorageBuffer, bufferSize,
            "ModelInstanceTransformRing");
    }

    {
        const uint32_t bufferSize =
            (DirectionalLight::sBufferByteSize + RingBuffer::sAlignment +
             PointLights::sBufferByteSize + RingBuffer::sAlignment +
             SpotLights::sBufferByteSize + RingBuffer::sAlignment) *
            MAX_FRAMES_IN_FLIGHT;
        _lightDataRing = std::make_unique<RingBuffer>(
            _device, vk::BufferUsageFlagBits::eStorageBuffer, bufferSize,
            "LightDataRing");
    }
}

void World::reflectBindings(ScopedScratch scopeAlloc)
{
    const auto reflect =
        [&](const String &defines, const std::filesystem::path &relPath)
    {
        Optional<ShaderReflection> compResult = _device->reflectShader(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = relPath,
                .defines = defines,
            },
            true);
        if (!compResult.has_value())
            throw std::runtime_error(
                std::string("Failed to create reflection for '") +
                relPath.string() + '\'');

        return WHEELS_MOV(*compResult);
    };

    {
        assert(!_samplers.empty());
        _dsLayouts.materialSamplerCount =
            asserted_cast<uint32_t>(_samplers.size());

        const size_t len = 192;
        String defines{scopeAlloc, len};
        appendDefineStr(
            defines, "MATERIAL_DATAS_SET", sMaterialDatasReflectionSet);
        appendDefineStr(
            defines, "MATERIAL_TEXTURES_SET", sMaterialTexturesReflectionSet);
        appendDefineStr(
            defines, "NUM_MATERIAL_SAMPLERS", _dsLayouts.materialSamplerCount);
        defines.extend("#extension GL_EXT_nonuniform_qualifier : require\n");
        assert(defines.size() <= len);

        _materialsReflection = reflect(defines, "shader/scene/materials.glsl");
    }

    {
        const size_t len = 92;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "GEOMETRY_SET", sGeometryReflectionSet);
        defines.extend("#extension GL_EXT_nonuniform_qualifier : require\n");
        assert(defines.size() <= len);

        _geometryReflection = reflect(defines, "shader/scene/geometry.glsl");
    }

    {
        const size_t len = 64;
        String defines{scopeAlloc, len};
        appendDefineStr(
            defines, "MODEL_INSTANCE_TRFNS_SET", sInstanceTrfnsReflectionSet);
        assert(defines.size() <= len);

        _modelInstancesReflection =
            reflect(defines, "shader/scene/transforms.glsl");
    }

    {
        const size_t len = 92;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "LIGHTS_SET", sLightsReflectionSet);
        PointLights::appendShaderDefines(defines);
        SpotLights::appendShaderDefines(defines);
        assert(defines.size() <= len);

        _lightsReflection = reflect(defines, "shader/scene/lights.glsl");
    }

    {
        const size_t len = 32;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "SKYBOX_SET", sSkyboxReflectionSet);
        assert(defines.size() <= len);

        _skyboxReflection = reflect(defines, "shader/scene/skybox.glsl");
    }
}

void World::createDescriptorSets(ScopedScratch scopeAlloc)
{
    if (_device == nullptr)
        throw std::runtime_error(
            "Tried to create World descriptor sets before loading glTF");

    assert(_materialsReflection.has_value());
    _dsLayouts.materialDatas = _materialsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), *_device, sMaterialDatasReflectionSet,
        vk::ShaderStageFlagBits::eFragment |
            vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eAnyHitKHR);

    {
        StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT>
            materialDatasLayouts;
        materialDatasLayouts.resize(
            MAX_FRAMES_IN_FLIGHT, _dsLayouts.materialDatas);
        _materialDatasDSs.resize(MAX_FRAMES_IN_FLIGHT);
        _descriptorAllocator.allocate(materialDatasLayouts, _materialDatasDSs);
    }

    assert(_materialsBuffers.size() == MAX_FRAMES_IN_FLIGHT);
    assert(_materialDatasDSs.size() == MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        const StaticArray descriptorInfos = {
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _materialsBuffers[i].handle,
                .range = VK_WHOLE_SIZE,
            }},
        };
        const StaticArray descriptorWrites =
            _materialsReflection->generateDescriptorWrites(
                sMaterialDatasReflectionSet, _materialDatasDSs[i],
                descriptorInfos);
        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    {
        Array<vk::DescriptorImageInfo> materialSamplerInfos{
            scopeAlloc, _samplers.size()};
        for (const auto &s : _samplers)
            materialSamplerInfos.push_back(
                vk::DescriptorImageInfo{.sampler = s});
        const auto samplerInfoCount =
            asserted_cast<uint32_t>(materialSamplerInfos.size());
        _dsLayouts.materialSamplerCount = samplerInfoCount;

        // Use capacity instead of size so that this allocates descriptors for
        // textures that are loaded later
        Array<vk::DescriptorImageInfo> materialImageInfos{
            scopeAlloc, _texture2Ds.capacity()};
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

        const StaticArray bindingFlags{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
                // Texture bindings for deferred loads are updated before frame
                // cb submission, for textures that aren't accessed by any frame
                // in flight
                vk::DescriptorBindingFlagBits::ePartiallyBound |
                vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending},
        };

        assert(_materialsReflection.has_value());
        _dsLayouts.materialTextures =
            _materialsReflection->createDescriptorSetLayout(
                scopeAlloc.child_scope(), *_device,
                sMaterialTexturesReflectionSet,
                vk::ShaderStageFlagBits::eFragment |
                    vk::ShaderStageFlagBits::eRaygenKHR |
                    vk::ShaderStageFlagBits::eAnyHitKHR,
                Span{&imageInfoCount, 1}, bindingFlags);

        _materialTexturesDS = _descriptorAllocator.allocate(
            _dsLayouts.materialTextures, imageInfoCount);

        const StaticArray descriptorInfos{
            DescriptorInfo{materialSamplerInfos},
            DescriptorInfo{materialImageInfos},
        };

        const StaticArray descriptorWrites =
            _materialsReflection->generateDescriptorWrites(
                sMaterialTexturesReflectionSet, _materialTexturesDS,
                descriptorInfos);
        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);

        if (_deferredLoadingContext.has_value())
            _deferredLoadingContext->textureArrayBinding =
                asserted_cast<uint32_t>(materialSamplerInfos.size());
    }

    {
        // Geometry layouts and descriptor set
        Array<vk::DescriptorBufferInfo> bufferInfos{
            scopeAlloc, 1 + _geometryBuffers.size()};

        bufferInfos.push_back(vk::DescriptorBufferInfo{
            .buffer = _meshBuffersBuffer.handle,
            .range = VK_WHOLE_SIZE,
        });

        for (const auto &b : _geometryBuffers)
            bufferInfos.push_back(vk::DescriptorBufferInfo{
                .buffer = b.handle,
                .range = VK_WHOLE_SIZE,
            });
        const auto bufferCount =
            asserted_cast<uint32_t>(bufferInfos.size() - 1);

        const StaticArray bindingFlags = {
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount},
        };

        assert(_geometryReflection.has_value());
        _dsLayouts.geometry = _geometryReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), *_device, sGeometryReflectionSet,
            vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eRaygenKHR |
                vk::ShaderStageFlagBits::eAnyHitKHR,
            Span{&bufferCount, 1}, bindingFlags);

        _geometryDS =
            _descriptorAllocator.allocate(_dsLayouts.geometry, bufferCount);

        const StaticArray descriptorInfos{
            DescriptorInfo{bufferInfos[0]},
            DescriptorInfo{bufferInfos.span(1, bufferInfos.size())},
        };

        const StaticArray descriptorWrites =
            _geometryReflection->generateDescriptorWrites(
                sGeometryReflectionSet, _geometryDS, descriptorInfos);

        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    // RT layout
    {
        // TODO:
        // Need to support differing flags for binds within set here? Does AMD
        // support binding AS in stages other than raygen (recursion = 1)? Is
        // perf affected if AS is bound but unused in anyhit?
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

    assert(_modelInstancesReflection.has_value());
    _dsLayouts.modelInstances =
        _modelInstancesReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), *_device, sInstanceTrfnsReflectionSet,
            vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eRaygenKHR |
                vk::ShaderStageFlagBits::eAnyHitKHR);

    assert(_lightsReflection.has_value());
    _dsLayouts.lights = _lightsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), *_device, sLightsReflectionSet,
        vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eRaygenKHR);

    // Per light type
    {
        _lightsDescriptorSet = _descriptorAllocator.allocate(_dsLayouts.lights);

        const StaticArray lightInfos{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _lightDataRing->buffer(),
                .offset = 0,
                .range = sizeof(DirectionalLight::Parameters),
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _lightDataRing->buffer(),
                .offset = 0,
                .range = PointLights::sBufferByteSize,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _lightDataRing->buffer(),
                .offset = 0,
                .range = SpotLights::sBufferByteSize,
            }},
        };

        const StaticArray descriptorWrites =
            _lightsReflection->generateDescriptorWrites(
                sLightsReflectionSet, _lightsDescriptorSet, lightInfos);

        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    // Scene descriptor sets
    for (auto &scene : _scenes)
    {
        {
            scene.modelInstancesDescriptorSet =
                _descriptorAllocator.allocate(_dsLayouts.modelInstances);

            const StaticArray descriptorInfos{
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _modelInstanceTransformsRing->buffer(),
                    .range = scene.modelInstances.size() *
                             sizeof(ModelInstance::Transforms),
                }},
            };
            const StaticArray descriptorWrites =
                _modelInstancesReflection->generateDescriptorWrites(
                    sInstanceTrfnsReflectionSet,
                    scene.modelInstancesDescriptorSet, descriptorInfos);

            _device->logical().updateDescriptorSets(
                asserted_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(), 0, nullptr);
        }
        {
            scene.rtDescriptorSet =
                _descriptorAllocator.allocate(_dsLayouts.rayTracing);

            const vk::DescriptorBufferInfo instanceInfo{
                .buffer = scene.rtInstancesBuffer.handle,
                .range = VK_WHOLE_SIZE};

            StaticArray descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet = scene.rtDescriptorSet,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType =
                        vk::DescriptorType::eAccelerationStructureKHR,
                },
                vk::WriteDescriptorSet{
                    .dstSet = scene.rtDescriptorSet,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &instanceInfo,
                },
            };

            // TODO:
            // This seems potentially messy to support with the
            // common reflection interface
            const vk::WriteDescriptorSetAccelerationStructureKHR asWrite{
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &_tlases[0].handle,
            };
            descriptorWrites[0].pNext = &asWrite;

            _device->logical().updateDescriptorSets(
                asserted_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(), 0, nullptr);
        }
    }

    // Skybox layout and descriptor set
    {
        assert(_skyboxReflection.has_value());
        _dsLayouts.skybox = _skyboxReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), *_device, sSkyboxReflectionSet,
            vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eRaygenKHR);

        _skyboxDS = _descriptorAllocator.allocate(_dsLayouts.skybox);

        const StaticArray descriptorInfos{
            DescriptorInfo{_skyboxTexture.imageInfo()},
        };
        const StaticArray descriptorWrites =
            _skyboxReflection->generateDescriptorWrites(
                sSkyboxReflectionSet, _skyboxDS, descriptorInfos);

        _device->logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }
}

bool World::pollTextureWorker(vk::CommandBuffer cb)
{
    assert(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    assert(ctx.loadedImageCount < ctx.gltfModel.images.size());

    bool newTextureLoaded = false;
    {
        const std::lock_guard _lock{ctx.loadedTextureMutex};
        if (ctx.loadedTexture.has_value())
        {
            _texture2Ds.emplace_back(ctx.loadedTexture.take());
            newTextureLoaded = true;
        }
    }

    if (newTextureLoaded)
    {
        ctx.loadedTextureTaken.notify_all();

        const QueueFamilies &families = _device->queueFamilies();
        assert(families.graphicsFamily.has_value());
        assert(families.transferFamily.has_value());

        if (*families.graphicsFamily != *families.transferFamily)
        {
            const vk::ImageMemoryBarrier2 acquireBarrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask =
                    vk::PipelineStageFlagBits2::eFragmentShader |
                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = *families.transferFamily,
                .dstQueueFamilyIndex = *families.graphicsFamily,
                .image = _texture2Ds.back().nativeHandle(),
                .subresourceRange =
                    vk::ImageSubresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &acquireBarrier,
            });
        }
    }

    return newTextureLoaded;
}

void World::loadTextureSingleThreaded(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame)
{
    assert(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    assert(ctx.loadedImageCount < ctx.gltfModel.images.size());

    assert(ctx.gltfModel.images.size() > ctx.loadedImageCount);
    const tinygltf::Image &image = ctx.gltfModel.images[ctx.loadedImageCount];
    if (image.uri.empty())
        throw std::runtime_error("Embedded glTF textures aren't "
                                 "supported. Scene should be glTF + "
                                 "bin + textures.");

    assert(ctx.stagingBuffers.size() > nextFrame);
    _texture2Ds.emplace_back(
        scopeAlloc.child_scope(), _device, _sceneDir / image.uri, cb,
        ctx.stagingBuffers[nextFrame], true);
}

void World::updateDescriptorsWithNewTexture()
{
    assert(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;

    const vk::DescriptorImageInfo imageInfo = _texture2Ds.back().imageInfo();
    const vk::WriteDescriptorSet descriptorWrite{
        .dstSet = _materialTexturesDS,
        .dstBinding = ctx.textureArrayBinding,
        // loadedImageCount is gltf images so bump by one to take our
        // default texture into account
        .dstArrayElement = ctx.loadedImageCount + 1,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eSampledImage,
        .pImageInfo = &imageInfo,
    };
    _device->logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);

    ctx.loadedImageCount++;

    // Update next material(s) in line if the required textures are
    // loaded
    bool materialsUpdated = false;
    for (size_t i = ctx.loadedMaterialCount; i < ctx.materials.size(); ++i)
    {
        const Material &material = ctx.materials[i];
        const uint32_t baseColorIndex = material.baseColor.texture();
        const uint32_t normalIndex = material.normal.texture();
        const uint32_t metallicRoughnessIndex =
            material.metallicRoughness.texture();
        // Inclusive as 0 is our default, starting gltf indices from 1
        if (baseColorIndex <= ctx.loadedImageCount &&
            normalIndex <= ctx.loadedImageCount &&
            metallicRoughnessIndex <= ctx.loadedImageCount)
        {
            // These are gltf material indices so we have to take our
            // default material into account
            _materials[i + 1] = material;
            ctx.loadedMaterialCount++;
            materialsUpdated = true;
        }
        else
            break;
    }

    if (materialsUpdated)
        ctx.materialsGeneration++;
}

World::DeferredLoadingContext::DeferredLoadingContext(
    Allocator &alloc, Device *device, const std::filesystem::path *sceneDir,
    const tinygltf::Model &gltfModel)
: device{device}
, gltfModel{gltfModel}
, materials{alloc, gltfModel.materials.size()}
{
    assert(sceneDir != nullptr);
    assert(device != nullptr);

    // One of these is used by the worker implementation, all by the
    // single threaded one
    for (uint32_t i = 0; i < stagingBuffers.capacity(); ++i)
        stagingBuffers.push_back(createTextureStaging(device));

    const Optional<vk::CommandPool> transferPool = device->transferPool();
    if (transferPool.has_value())
    {
        assert(device->transferQueue().has_value());

        cb = device->logical().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{
                .commandPool = *transferPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1})[0];
        worker = std::thread{&loadingWorker, sceneDir, this};
    }
}

World::DeferredLoadingContext::~DeferredLoadingContext()
{
    if (device != nullptr)
    {
        if (worker.has_value())
        {
            {
                const std::lock_guard _lock{loadedTextureMutex};
                if (loadedTexture.has_value())
                    const Texture2D _tex = loadedTexture.take();
            }
            loadedTextureTaken.notify_all();

            interruptLoading = true;
            worker->join();
        }

        for (const Buffer &buffer : stagingBuffers)
            device->destroy(buffer);
    }
}
