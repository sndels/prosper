#ifndef PROSPER_SCENE_WORLD_HPP
#define PROSPER_SCENE_WORLD_HPP

#include "../gfx/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"

#include <filesystem>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class World
{
  public:
    World(
        wheels::Allocator &generalAlloc, wheels::ScopedScratch scopeAlloc,
        Device *device, RingBuffer *constantsRing,
        const std::filesystem::path &scene);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    void startFrame();
    void endFrame();

    // Returns true if the visible scene was changed.
    bool handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    void drawDeferredLoadingUi() const;
    // Returns true if the next frame will use a different scene
    bool drawSceneUi();
    // Returns true if the active camera was changed
    [[nodiscard]] bool drawCameraUi();

    [[nodiscard]] Scene &currentScene();
    [[nodiscard]] const Scene &currentScene() const;

    [[nodiscard]] AccelerationStructure &currentTLAS();

    [[nodiscard]] CameraParameters const &currentCamera() const;
    [[nodiscard]] bool isCurrentCameraDynamic() const;

    void uploadMaterialDatas(uint32_t nextFrame, float lodBias);
    void updateAnimations(float timeS, Profiler *profiler);
    // Has to be called after updateAnimations()
    void updateScene(
        wheels::ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
        Profiler *profiler);
    void updateBuffers(wheels::ScopedScratch scopeAlloc);
    // Has to be called after updateBuffers(). Returns true if new BLASes were
    // added.
    bool buildAccelerationStructures(vk::CommandBuffer cb);
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
