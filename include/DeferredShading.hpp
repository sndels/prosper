#ifndef PROSPER_DEFERRED_SHADING_HPP
#define PROSPER_DEFERRED_SHADING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "ComputePass.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "GBufferRenderer.hpp"
#include "LightClustering.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"
#include "World.hpp"

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
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

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
        const Input &input, uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_DEFERRED_SHADING_HPP
