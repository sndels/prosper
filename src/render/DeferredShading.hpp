#ifndef PROSPER_RENDER_DEFERRED_SHADING_HPP
#define PROSPER_RENDER_DEFERRED_SHADING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../gfx/Fwd.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

class DeferredShading
{
  public:
    enum class DrawType : uint32_t
    {
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    DeferredShading() noexcept = default;
    ~DeferredShading() = default;

    DeferredShading(const DeferredShading &other) = delete;
    DeferredShading(DeferredShading &&other) = delete;
    DeferredShading &operator=(const DeferredShading &other) = delete;
    DeferredShading &operator=(DeferredShading &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    void drawUi();

    struct Input
    {
        const GBufferRendererOutput &gbuffer;
        const LightClusteringOutput &lightClusters;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const Input &input,
        uint32_t nextFrame, bool applyIbl, Profiler *profiler);

    bool _initialized{false};
    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDER_DEFERRED_SHADING_HPP
