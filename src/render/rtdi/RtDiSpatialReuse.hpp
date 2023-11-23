#ifndef PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP
#define PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Device.hpp"
#include "../../scene/Camera.hpp"
#include "../../scene/World.hpp"
#include "../../utils/Profiler.hpp"
#include "../ComputePass.hpp"
#include "../GBufferRenderer.hpp"
#include "../RenderResources.hpp"

class RtDiSpatialReuse
{
  public:
    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const World::DSLayouts &world;
    };
    RtDiSpatialReuse(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

    ~RtDiSpatialReuse() = default;

    RtDiSpatialReuse(const RtDiSpatialReuse &other) = delete;
    RtDiSpatialReuse(RtDiSpatialReuse &&other) = delete;
    RtDiSpatialReuse &operator=(const RtDiSpatialReuse &other) = delete;
    RtDiSpatialReuse &operator=(RtDiSpatialReuse &&other) = delete;

    // Returns true if recompile happened
    bool recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct Input
    {
        const GBufferRenderer::Output &gbuffer;
        ImageHandle reservoirs;
    };
    struct Output
    {
        ImageHandle reservoirs;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const Input &input, uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DI_SPATIAL_REUSE_HPP
