#ifndef PROSPER_SCENE_WORLD_HPP
#define PROSPER_SCENE_WORLD_HPP

#include "../gfx/RingBuffer.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Timer.hpp"
#include "Animations.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "Texture.hpp"

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

    void startFrame() const;

    void uploadMaterialDatas(uint32_t nextFrame);
    void handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    void drawDeferredLoadingUi() const;

    [[nodiscard]] Scene &currentScene();
    [[nodiscard]] const Scene &currentScene() const;
    void updateAnimations(float timeS, Profiler *profiler);
    // Has to be called after updateAnimations()
    void updateScene(wheels::ScopedScratch scopeAlloc, Profiler *profiler);
    void updateBuffers(wheels::ScopedScratch scopeAlloc);
    // Has to be called after updateBuffers()
    void buildCurrentTlas(vk::CommandBuffer cb);
    void drawSkybox(const vk::CommandBuffer &buffer) const;

    wheels::Allocator &_generalAlloc;
    wheels::LinearAllocator _linearAlloc;

    std::filesystem::path _sceneDir;

    // TODO: Private?
    TextureCubemap _skyboxTexture;
    Buffer _skyboxVertexBuffer;

    wheels::Array<CameraParameters> _cameras{_generalAlloc};
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
    wheels::Array<uint8_t> _rawAnimationData{_generalAlloc};
    Animations _animations{_generalAlloc};
    wheels::Array<Scene> _scenes{_generalAlloc};
    size_t _currentScene{0};
    uint32_t _currentCamera{0};

    DSLayouts _dsLayouts;

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _materialsBuffers;
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> _materialsGenerations{
        0};
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _materialDatasDSs;
    vk::DescriptorSet _materialTexturesDS;
    wheels::Optional<ShaderReflection> _materialsReflection;

    vk::DescriptorSet _geometryDS;
    wheels::Optional<ShaderReflection> _geometryReflection;

    wheels::Optional<ShaderReflection> _modelInstancesReflection;
    std::unique_ptr<RingBuffer> _modelInstanceTransformsRing;
    uint32_t _modelInstanceTransformsByteOffset{0};

    wheels::Optional<ShaderReflection> _lightsReflection;
    vk::DescriptorSet _lightsDescriptorSet;
    std::unique_ptr<RingBuffer> _lightDataRing;
    uint32_t _directionalLightByteOffset{0};
    uint32_t _pointLightByteOffset{0};
    uint32_t _spotLightByteOffset{0};

    wheels::Optional<ShaderReflection> _skyboxReflection;
    vk::DescriptorSet _skyboxDS;

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
        std::atomic<uint32_t> allocationHighWatermark{0};

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
    uint32_t _deferredLoadingAllocationHighWatermark{0};

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

    struct NodeAnimations
    {
        wheels::Optional<Animation<glm::vec3> *> translation;
        wheels::Optional<Animation<glm::quat> *> rotation;
        wheels::Optional<Animation<glm::vec3> *> scale;
    };
    wheels::HashMap<uint32_t, NodeAnimations> loadAnimations(
        wheels::Allocator &alloc, wheels::ScopedScratch scopeAlloc,
        const tinygltf::Model &gltfModel);

    void loadScenes(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
        const wheels::HashMap<uint32_t, NodeAnimations> &nodeAnimations);

    struct TmpNode
    {
        const std::string &gltfName;
        wheels::Array<uint32_t> children;
        wheels::Optional<glm::vec3> translation;
        wheels::Optional<glm::quat> rotation;
        wheels::Optional<glm::vec3> scale;
        wheels::Optional<uint32_t> modelID;
        wheels::Optional<uint32_t> camera;
        wheels::Optional<uint32_t> light;

        TmpNode(wheels::Allocator &alloc, const std::string &gltfName)
        : children{alloc}
        , gltfName{gltfName}
        {
        }
    };
    void gatherScene(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
        const tinygltf::Scene &gltfScene, const wheels::Array<TmpNode> &nodes);

    void createBlases();
    void createTlases(wheels::ScopedScratch scopeAlloc);
    void reserveScratch(vk::DeviceSize byteSize);
    void reserveTlasInstances(
        wheels::Span<const vk::AccelerationStructureInstanceKHR> instances);
    void updateTlasInstances(
        wheels::ScopedScratch scopeAlloc, const Scene &scene);
    void createTlasBuildInfos(
        const Scene &scene,
        vk::AccelerationStructureBuildRangeInfoKHR &rangeInfoOut,
        vk::AccelerationStructureGeometryKHR &geometryOut,
        vk::AccelerationStructureBuildGeometryInfoKHR &buildInfoOut,
        vk::AccelerationStructureBuildSizesInfoKHR &sizeInfoOut);
    void createBuffers();
    void reflectBindings(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);

    [[nodiscard]] bool pollTextureWorker(vk::CommandBuffer cb);

    void loadTextureSingleThreaded(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame);

    void updateDescriptorsWithNewTexture();

    Device *_device{nullptr};
    DescriptorAllocator _descriptorAllocator;

    Buffer _scratchBuffer;
    Buffer _tlasInstancesBuffer;
    std::unique_ptr<RingBuffer> _tlasInstancesUploadRing;
    uint32_t _tlasInstancesUploadOffset{0};
};

#endif // PROSPER_SCENE_WORLD_HPP
