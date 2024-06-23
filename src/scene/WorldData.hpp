#ifndef PROSPER_SCENE_WORLD_DATA_HPP
#define PROSPER_SCENE_WORLD_DATA_HPP

#include "../Allocators.hpp"
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
    WorldData() noexcept = default;
    ~WorldData();

    WorldData(const WorldData &) = delete;
    WorldData(WorldData &&) = delete;
    WorldData &operator=(const WorldData &) = delete;
    WorldData &operator=(WorldData &&) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, const RingBuffers &ringBuffers,
        const std::filesystem::path &scene);

    void uploadMeshDatas(wheels::ScopedScratch scopeAlloc, uint32_t nextFrame);
    void uploadMaterialDatas(uint32_t nextFrame);
    // Returns true if the visible scene was changed.
    bool handleDeferredLoading(vk::CommandBuffer cb, Profiler *profiler);

    void drawDeferredLoadingUi() const;

  private:
    bool _initialized{false};
    // Use general for descriptors because because we don't know the required
    // storage up front and the internal array will be reallocated
    DescriptorAllocator _descriptorAllocator;

    Timer _materialStreamingTimer;

    std::filesystem::path _sceneDir;

    wheels::Array<vk::Sampler> _samplers{gAllocators.general};
    wheels::Array<Texture2D> _texture2Ds{gAllocators.general};
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _geometryMetadatasBuffers;
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _meshletCountsBuffers;
    wheels::Array<uint32_t> _geometryBufferAllocatedByteCounts{
        gAllocators.general};
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> _geometryGenerations{0};

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> _materialsBuffers;
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> _materialsGenerations{
        0};

    wheels::Array<uint8_t> _rawAnimationData{gAllocators.general};

    wheels::Optional<ShaderReflection> _materialsReflection;
    wheels::Optional<ShaderReflection> _geometryReflection;
    wheels::Optional<ShaderReflection> _sceneInstancesReflection;
    wheels::Optional<ShaderReflection> _lightsReflection;
    wheels::Optional<ShaderReflection> _skyboxReflection;

  public:
    SkyboxResources _skyboxResources{
        .radianceViews = wheels::Array<vk::ImageView>{gAllocators.general},
    };

    wheels::Array<CameraParameters> _cameras{gAllocators.general};
    // True if any instance of the camera is dynamic
    wheels::Array<bool> _cameraDynamic{gAllocators.general};
    wheels::Array<Material> _materials{gAllocators.general};
    wheels::Array<Buffer> _geometryBuffers{gAllocators.general};
    wheels::Array<GeometryMetadata> _geometryMetadatas{gAllocators.general};
    wheels::Array<MeshInfo> _meshInfos{gAllocators.general};
    wheels::Array<wheels::String> _meshNames{gAllocators.general};
    wheels::Array<AccelerationStructure> _blases{gAllocators.general};
    wheels::Array<AccelerationStructure> _tlases{gAllocators.general};
    wheels::Array<Model> _models{gAllocators.general};
    Animations _animations{gAllocators.general};
    wheels::Array<Scene> _scenes{gAllocators.general};
    size_t _currentScene{0};

    WorldDSLayouts _dsLayouts;
    WorldDescriptorSets _descriptorSets;

    RingBuffer _modelInstanceTransformsRing;

    wheels::Optional<DeferredLoadingContext> _deferredLoadingContext;

  private:
    void loadTextures(
        wheels::ScopedScratch scopeAlloc, const cgltf_data &gltfData,
        wheels::Array<Texture2DSampler> &texture2DSamplers);
    void loadMaterials(
        const cgltf_data &gltfData,
        const wheels::Array<Texture2DSampler> &texture2DSamplers);
    void loadModels(
        wheels::ScopedScratch scopeAlloc, const cgltf_data &gltfData);

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
        wheels::ScopedScratch scopeAlloc, const cgltf_data &gltfData);

    void loadScenes(
        wheels::ScopedScratch scopeAlloc, const cgltf_data &gltfData,
        const wheels::HashMap<uint32_t, NodeAnimations> &nodeAnimations);

    struct TmpNode
    {
        wheels::String gltfName;
        wheels::Array<uint32_t> children;
        wheels::Optional<glm::vec3> translation;
        wheels::Optional<glm::quat> rotation;
        wheels::Optional<glm::vec3> scale;
        wheels::Optional<uint32_t> modelID;
        wheels::Optional<uint32_t> camera;
        wheels::Optional<uint32_t> light;

        TmpNode(const char *gltfName)
        : gltfName{gAllocators.general, gltfName}
        , children{gAllocators.general}
        {
        }
    };
    void gatherScene(
        wheels::ScopedScratch scopeAlloc, const cgltf_data &gltfData,
        const cgltf_scene &gltfScene, const wheels::Array<TmpNode> &nodes);

    void createBlases();
    void createBuffers();
    void reflectBindings(wheels::ScopedScratch scopeAlloc);
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc, const RingBuffers &ringBuffers);

    [[nodiscard]] bool pollMeshWorker(vk::CommandBuffer cb);
    // Returns the count of newly loaded textures
    [[nodiscard]] size_t pollTextureWorker(vk::CommandBuffer cb);

    void updateDescriptorsWithNewTextures(size_t newTextureCount);
    bool updateMaterials();
};

#endif // PROSPER_SCENE_WORLD_DATA_HPP
