#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Profiler.hpp"
#include "Scene.hpp"
#include "Texture.hpp"
#include "Timer.hpp"

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include <tiny_gltf.h>
#include <vulkan/vulkan_hash.hpp>

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/hash_map.hpp>

namespace tinygltf
{
class Model;
};

class Device;
struct DeferredLoadingContext;

class World
{
  public:
    struct DSLayouts
    {
        uint32_t materialSamplerCount{0};
        vk::DescriptorSetLayout materialDatas;
        vk::DescriptorSetLayout materialTextures;
        vk::DescriptorSetLayout geometry;
        vk::DescriptorSetLayout modelInstances;
        vk::DescriptorSetLayout rayTracing;
        vk::DescriptorSetLayout lights;
        vk::DescriptorSetLayout skybox;
        vk::DescriptorSetLayout skyboxOnly;
    };
    World(
        wheels::Allocator &generalAlloc, wheels::ScopedScratch scopeAlloc,
        Device *device, const std::filesystem::path &scene,
        bool deferredLoading);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    void uploadMaterialDatas(uint32_t nextFrame);
    void handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    void drawDeferredLoadingUi() const;

    [[nodiscard]] const Scene &currentScene() const;
    void updateUniformBuffers(
        const Camera &cam, uint32_t nextFrame,
        wheels::ScopedScratch scopeAlloc) const;
    void drawSkybox(const vk::CommandBuffer &buffer) const;

    wheels::Allocator &_generalAlloc;
    wheels::LinearAllocator _linearAlloc;

    std::filesystem::path _sceneDir;

    // TODO: Private?
    TextureCubemap _skyboxTexture;
    Buffer _skyboxVertexBuffer;

    wheels::HashMap<Scene::Node *, CameraParameters> _cameras{_generalAlloc};
    wheels::Array<vk::Sampler> _samplers{_generalAlloc};
    wheels::Array<Texture2D> _texture2Ds{_generalAlloc};
    wheels::Array<Buffer> _geometryBuffers{_generalAlloc};
    Buffer _meshBuffersBuffer;
    wheels::Array<Material> _materials{_generalAlloc};
    wheels::Array<MeshBuffers> _meshBuffers{_generalAlloc};
    wheels::Array<MeshInfo> _meshInfos{_generalAlloc};
    wheels::Array<Buffer> _modelInstances{_generalAlloc};
    wheels::Array<AccelerationStructure> _blases{_generalAlloc};
    wheels::Array<AccelerationStructure> _tlases{_generalAlloc};
    wheels::Array<Model> _models{_generalAlloc};
    wheels::Array<Scene::Node> _nodes{_generalAlloc};
    wheels::Array<Scene> _scenes{_generalAlloc};
    size_t _currentScene{0};

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _materialsBuffers;
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> _materialsGenerations{
        0};
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _materialDatasDSs;
    vk::DescriptorSet _materialTexturesDS;

    vk::DescriptorSet _geometryDS;
    DSLayouts _dsLayouts;

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _skyboxUniformBuffers;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> _skyboxDSs;
    vk::DescriptorSet _skyboxOnlyDS;

    struct DeferredLoadingContext
    {
        DeferredLoadingContext(
            wheels::Allocator &alloc, Device *device,
            const std::filesystem::path *sceneDir,
            const tinygltf::Model &gltfModel);
        ~DeferredLoadingContext();

        DeferredLoadingContext(const DeferredLoadingContext &) = delete;
        DeferredLoadingContext(DeferredLoadingContext &&) = delete;
        DeferredLoadingContext &operator=(const DeferredLoadingContext &) =
            delete;
        DeferredLoadingContext &operator=(DeferredLoadingContext &&) = delete;

        Device *device{nullptr};
        // If there's no worker, main thread handles loading
        wheels::Optional<std::thread> worker;

        // Worker context
        tinygltf::Model gltfModel;
        vk::CommandBuffer cb;
        uint32_t workerLoadedImageCount{0};

        // Shared context
        std::mutex loadedTextureMutex;
        std::condition_variable loadedTextureTaken;
        wheels::Optional<Texture2D> loadedTexture;
        std::atomic<bool> interruptLoading{false};

        // Main context
        uint32_t materialsGeneration{0};
        uint32_t framesSinceFinish{0};
        uint32_t textureArrayBinding{0};
        uint32_t loadedImageCount{0};
        uint32_t loadedMaterialCount{0};
        wheels::Array<Material> materials;
        wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> stagingBuffers;
        Timer timer;
    };
    wheels::Optional<DeferredLoadingContext> _deferredLoadingContext;

  private:
    void loadTextures(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
        wheels::Array<Texture2DSampler> &texture2DSamplers,
        bool deferredLoading);
    void loadMaterials(
        const tinygltf::Model &gltfModel,
        const wheels::Array<Texture2DSampler> &texture2DSamplers,
        bool deferredLoading);
    void loadModels(const tinygltf::Model &gltfModel);
    void loadScenes(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel);
    void createBlases();
    void createTlases(wheels::ScopedScratch scopeAlloc);
    void createBuffers();
    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);

    [[nodiscard]] bool pollTextureWorker(vk::CommandBuffer cb);

    void loadTextureSingleThreaded(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame);

    void updateDescriptorsWithNewTexture();

    Device *_device{nullptr};
    DescriptorAllocator _descriptorAllocator;
};

#endif // PROSPER_WORLD_HPP
