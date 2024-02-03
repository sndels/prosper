#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldGather
{
  public:
    enum GatherType : uint32_t
    {
        GatherType_Foreground,
        GatherType_Background,
        GatherType_Count,
    };

    DepthOfFieldGather() noexcept = default;
    ~DepthOfFieldGather() = default;

    DepthOfFieldGather(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather(DepthOfFieldGather &&other) = delete;
    DepthOfFieldGather &operator=(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather &operator=(DepthOfFieldGather &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Input
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCoC;
        ImageHandle dilatedTileMinMaxCoC;
    };
    struct Output
    {
        ImageHandle halfResBokehColorWeight;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, GatherType gatherType, uint32_t nextFrame,
        Profiler *profiler);

    bool _initialized{false};
    RenderResources *_resources{nullptr};

    ComputePass _backgroundPass;
    ComputePass _foregroundPass;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
