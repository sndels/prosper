#ifndef PROSPER_SCENE_WORLD_HPP
#define PROSPER_SCENE_WORLD_HPP

#include "../gfx/Fwd.hpp"
#include "../scene/Texture.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"

#include <filesystem>
#include <memory>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

struct WorldDSLayouts
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

struct WorldByteOffsets
{
    uint32_t modelInstanceTransforms{0};
    uint32_t previousModelInstanceTransforms{0};
    uint32_t directionalLight{0};
    uint32_t pointLights{0};
    uint32_t spotLights{0};
};

struct WorldDescriptorSets
{
    vk::DescriptorSet lights;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> materialDatas;
    vk::DescriptorSet materialTextures;
    vk::DescriptorSet geometry;
    vk::DescriptorSet skybox;
};

struct SkyboxResources
{
    TextureCubemap texture;
    Image irradiance;
    Image specularBrdfLut;
    Image radiance;
    wheels::Array<vk::ImageView> radianceViews;
    Buffer vertexBuffer;
    vk::Sampler sampler;
};

class World
{
  public:
    static const uint32_t sSkyboxIrradianceResolution = 64;
    static const uint32_t sSpecularBrdfLutResolution = 512;
    static const uint32_t sSkyboxRadianceResolution = 512;

    World(
        wheels::Allocator &generalAlloc, wheels::ScopedScratch scopeAlloc,
        Device *device, const std::filesystem::path &scene,
        bool deferredLoading);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    void startFrame();
    void endFrame();

    void handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    void drawDeferredLoadingUi() const;
    // Returns true if the next frame will use a different scene
    bool drawSceneUi();
    // Returns true if the active camera was changed
    [[nodiscard]] bool drawCameraUi();

    [[nodiscard]] Scene &currentScene();
    [[nodiscard]] const Scene &currentScene() const;

    [[nodiscard]] CameraParameters const &currentCamera() const;
    [[nodiscard]] bool isCurrentCameraDynamic() const;

    void uploadMaterialDatas(uint32_t nextFrame);
    void updateAnimations(float timeS, Profiler *profiler);
    // Has to be called after updateAnimations()
    void updateScene(
        wheels::ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
        Profiler *profiler);
    void updateBuffers(wheels::ScopedScratch scopeAlloc);
    // Has to be called after updateBuffers()
    void buildCurrentTlas(vk::CommandBuffer cb);
    void drawSkybox(vk::CommandBuffer cb) const;

    [[nodiscard]] const WorldDSLayouts &dsLayouts() const;
    [[nodiscard]] const WorldDescriptorSets &descriptorSets() const;
    [[nodiscard]] const WorldByteOffsets &byteOffsets() const;
    [[nodiscard]] wheels::Span<const Model> models() const;
    [[nodiscard]] wheels::Span<const Material> materials() const;
    [[nodiscard]] wheels::Span<const MeshInfo> meshInfos() const;
    [[nodiscard]] SkyboxResources &skyboxResources();

    [[nodiscard]] size_t deferredLoadingAllocatorHighWatermark() const;
    [[nodiscard]] size_t linearAllocatorHighWatermark() const;

  private:
    // Pimpl to isolate heavy includes within the World CU
    class Impl;
    std::unique_ptr<Impl> _impl;
};

#endif // PROSPER_SCENE_WORLD_HPP
