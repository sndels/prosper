#ifndef PROSPER_SCENE_WORLD_DATA_HPP
#define PROSPER_SCENE_WORLD_DATA_HPP

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/Fwd.hpp"
#include "../gfx/RingBuffer.hpp"
#include "../utils/Profiler.hpp"
#include "Animations.hpp"
#include "Camera.hpp"
#include "DeferredLoadingContext.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "WorldRenderStructs.hpp"
#include <cstdint>
#include <memory>
#include <wheels/allocators/allocator.hpp>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/hash_map.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

// This implements loading and is used 'internally' by the world pimpl
class WorldData
{
  public:
    static const size_t sSkyboxVertsCount = 36;

    struct RingBuffers
    {
        RingBuffer *constantsRing{nullptr};
        RingBuffer *lightDataRing{nullptr};
    };
    WorldData(
        wheels::Allocator &generalAlloc, wheels::ScopedScratch scopeAlloc,
        Device *device, const RingBuffers &ringBuffers,
        const std::filesystem::path &scene, bool deferredLoading);
    ~WorldData();

    WorldData(const WorldData &) = delete;
    WorldData(WorldData &&) = delete;
    WorldData &operator=(const WorldData &) = delete;
    WorldData &operator=(WorldData &&) = delete;

    void uploadMaterialDatas(uint32_t nextFrame);
    void handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    void drawDeferredLoadingUi() const;
    [[nodiscard]] size_t linearAllocatorHighWatermark() const;

  private:
    wheels::Allocator &_generalAlloc;
    wheels::LinearAllocator _linearAlloc;
    Device *_device{nullptr};
    DescriptorAllocator _descriptorAllocator;
    Buffer _scratchBuffer;

    std::filesystem::path _sceneDir;

    wheels::Optional<DeferredLoadingContext> _deferredLoadingContext;

    wheels::Array<vk::Sampler> _samplers{_generalAlloc};
    wheels::Array<Texture2D> _texture2Ds{_generalAlloc};
    Buffer _geometryUploadBuffer;
    wheels::Array<Buffer> _geometryBuffers{_generalAlloc};
    wheels::Array<uint32_t> _geometryBufferRemainingByteCounts{_generalAlloc};
    Buffer _meshBuffersBuffer;
    wheels::Array<MeshBuffers> _meshBuffers{_generalAlloc};

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _materialsBuffers;
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> _materialsGenerations{
        0};

    wheels::Array<uint8_t> _rawAnimationData{_generalAlloc};

    wheels::Optional<ShaderReflection> _materialsReflection;
    wheels::Optional<ShaderReflection> _geometryReflection;
    wheels::Optional<ShaderReflection> _modelInstancesReflection;
    wheels::Optional<ShaderReflection> _lightsReflection;
    wheels::Optional<ShaderReflection> _skyboxReflection;

  public:
    SkyboxResources _skyboxResources;

    wheels::Array<CameraParameters> _cameras{_generalAlloc};
    // True if any instance of the camera is dynamic
    wheels::Array<bool> _cameraDynamic{_generalAlloc};
    wheels::Array<Material> _materials{_generalAlloc};
    wheels::Array<MeshInfo> _meshInfos{_generalAlloc};
    wheels::Array<AccelerationStructure> _blases{_generalAlloc};
    wheels::Array<AccelerationStructure> _tlases{_generalAlloc};
    wheels::Array<Model> _models{_generalAlloc};
    Animations _animations{_generalAlloc};
    wheels::Array<Scene> _scenes{_generalAlloc};
    size_t _currentScene{0};

    WorldDSLayouts _dsLayouts;
    WorldDescriptorSets _descriptorSets;

    std::unique_ptr<RingBuffer> _modelInstanceTransformsRing;

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

    struct InputBuffer
    {
        uint32_t index{0xFFFFFFFF};
        uint32_t byteOffset{0};
    };
    struct InputMeshBuffers
    {
        InputBuffer indices;
        InputBuffer positions;
        InputBuffer normals;
        InputBuffer tangents;
        InputBuffer texCoord0s;
        bool usesShortIndices{false};
    };
    MeshBuffers uploadMeshData(
        const tinygltf::Model &gltfModel, const InputMeshBuffers &meshBuffers,
        const MeshInfo &meshInfo);

    struct NodeAnimations
    {
        wheels::Optional<Animation<glm::vec3> *> translation;
        wheels::Optional<Animation<glm::quat> *> rotation;
        wheels::Optional<Animation<glm::vec3> *> scale;
        // TODO:
        // Dynamic light parameters. glTF doesn't have an official extension
        // that supports this so requires a custom exporter plugin fork.
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
        : gltfName{gltfName}
        , children{alloc}
        {
        }
    };
    void gatherScene(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
        const tinygltf::Scene &gltfScene, const wheels::Array<TmpNode> &nodes);

    void createBlases();
    void createBuffers();
    void reflectBindings(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc, const RingBuffers &ringBuffers);

    // Returns the count of newly loaded textures
    [[nodiscard]] size_t pollTextureWorker(vk::CommandBuffer cb);

    void loadTextureSingleThreaded(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame);

    void updateDescriptorsWithNewTextures(size_t newTextureCount);
};

#endif // PROSPER_SCENE_WORLD_DATA_HPP
