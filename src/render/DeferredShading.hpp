#ifndef PROSPER_RENDER_DEFERRED_SHADING_HPP
#define PROSPER_RENDER_DEFERRED_SHADING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Device.hpp"
#include "../gfx/Swapchain.hpp"
#include "../scene/Camera.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "GBufferRenderer.hpp"
#include "LightClustering.hpp"
#include "RenderResources.hpp"

class DeferredShading
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const World::DSLayouts &world;
    };
    DeferredShading(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

    ~DeferredShading() = default;

    DeferredShading(const DeferredShading &other) = delete;
    DeferredShading(DeferredShading &&other) = delete;
    DeferredShading &operator=(const DeferredShading &other) = delete;
    DeferredShading &operator=(DeferredShading &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    void drawUi();

    struct Input
    {
        const GBufferRenderer::Output &gbuffer;
        const LightClustering::Output &lightClusters;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const Input &input, uint32_t nextFrame, bool applyIbl,
        Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDER_DEFERRED_SHADING_HPP
