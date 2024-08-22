#ifndef PROSPER_SCENE_WORLD_DATA_HPP
#define PROSPER_SCENE_WORLD_DATA_HPP

#include "Allocators.hpp"
#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Fwd.hpp"
#include "gfx/RingBuffer.hpp"
#include "scene/Animations.hpp"
#include "scene/Camera.hpp"
#include "scene/DeferredLoadingContext.hpp"
#include "scene/Material.hpp"
#include "scene/Mesh.hpp"
#include "scene/Model.hpp"
#include "scene/Scene.hpp"
#include "scene/WorldRenderStructs.hpp"

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
    bool handleDeferredLoading(vk::CommandBuffer cb);

    void drawDeferredLoadingUi() const;

  private:
    bool m_initialized{false};
    // Use general for descriptors because because we don't know the required
    // storage up front and the internal array will be reallocated
    DescriptorAllocator m_descriptorAllocator;

    Timer m_materialStreamingTimer;

    std::filesystem::path m_sceneDir;

    wheels::Array<vk::Sampler> m_samplers{gAllocators.general};
    wheels::Array<Texture2D> m_texture2Ds{gAllocators.general};
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT>
        m_geometryMetadatasBuffers;
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> m_meshletCountsBuffers;
    wheels::Array<uint32_t> m_geometryBufferAllocatedByteCounts{
        gAllocators.general};
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> m_geometryGenerations{
        0};

    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> m_materialsBuffers;
    wheels::StaticArray<uint32_t, MAX_FRAMES_IN_FLIGHT> m_materialsGenerations{
        0};

    wheels::Array<uint8_t> m_rawAnimationData{gAllocators.general};

    wheels::Optional<ShaderReflection> m_materialsReflection;
    wheels::Optional<ShaderReflection> m_geometryReflection;
    wheels::Optional<ShaderReflection> m_sceneInstancesReflection;
    wheels::Optional<ShaderReflection> m_lightsReflection;
    wheels::Optional<ShaderReflection> m_skyboxReflection;

  public:
    SkyboxResources m_skyboxResources{
        .radianceViews = wheels::Array<vk::ImageView>{gAllocators.general},
    };

    wheels::Array<CameraParameters> m_cameras{gAllocators.general};
    // True if any instance of the camera is dynamic
    wheels::Array<bool> m_cameraDynamic{gAllocators.general};
    wheels::Array<Material> m_materials{gAllocators.general};
    wheels::Array<Buffer> m_geometryBuffers{gAllocators.general};
    wheels::Array<GeometryMetadata> m_geometryMetadatas{gAllocators.general};
    wheels::Array<MeshInfo> m_meshInfos{gAllocators.general};
    wheels::Array<wheels::String> m_meshNames{gAllocators.general};
    wheels::Array<AccelerationStructure> m_blases{gAllocators.general};
    wheels::Array<AccelerationStructure> m_tlases{gAllocators.general};
    wheels::Array<Model> m_models{gAllocators.general};
    Animations m_animations{gAllocators.general};
    wheels::Array<Scene> m_scenes{gAllocators.general};
    size_t m_currentScene{0};

    WorldDSLayouts m_dsLayouts;
    WorldDescriptorSets m_descriptorSets;

    RingBuffer m_modelInstanceTransformsRing;

    wheels::Optional<DeferredLoadingContext> m_deferredLoadingContext;

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
        wheels::Optional<uint32_t> modelIndex;
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
