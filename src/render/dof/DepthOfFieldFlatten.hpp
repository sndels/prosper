#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Device.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../RenderResources.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldFlatten
{
  public:
    DepthOfFieldFlatten(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldFlatten() = default;

    DepthOfFieldFlatten(const DepthOfFieldFlatten &other) = delete;
    DepthOfFieldFlatten(DepthOfFieldFlatten &&other) = delete;
    DepthOfFieldFlatten &operator=(const DepthOfFieldFlatten &other) = delete;
    DepthOfFieldFlatten &operator=(DepthOfFieldFlatten &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Output
    {
        ImageHandle tileMinMaxCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle halfResCircleOfConfusion,
        uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP