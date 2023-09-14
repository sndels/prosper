#ifndef PROSPER_DEPTH_OF_FIELD_GATHER_HPP
#define PROSPER_DEPTH_OF_FIELD_GATHER_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "ComputePass.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldGather
{
  public:
    enum GatherType : uint32_t
    {
        GatherType_Foreground = 0,
        GatherType_Background = 1,
        GatherType_Count,
    };

    DepthOfFieldGather(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldGather() = default;

    DepthOfFieldGather(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather(DepthOfFieldGather &&other) = delete;
    DepthOfFieldGather &operator=(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather &operator=(DepthOfFieldGather &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

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
        vk::CommandBuffer cb, const Input &input, GatherType gatherType,
        uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};

    ComputePass _backgroundPass;
    ComputePass _foregroundPass;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_DEPTH_OF_FIELD_GATHER_HPP
