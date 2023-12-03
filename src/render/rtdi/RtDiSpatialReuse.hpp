#ifndef PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP
#define PROSPER_RENDER_RTDI_SPATIAL_REUSE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

class RtDiSpatialReuse
{
  public:
    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        const WorldDSLayouts &world;
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
        const GBufferRendererOutput &gbuffer;
        ImageHandle reservoirs;
    };
    struct Output
    {
        ImageHandle reservoirs;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const Input &input,
        uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DI_SPATIAL_REUSE_HPP
