#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP

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

class DepthOfFieldDilate
{
  public:
    DepthOfFieldDilate(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldDilate() = default;

    DepthOfFieldDilate(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate(DepthOfFieldDilate &&other) = delete;
    DepthOfFieldDilate &operator=(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate &operator=(DepthOfFieldDilate &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Output
    {
        ImageHandle dilatedTileMinMaxCoC;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle tileMinMaxCoC, uint32_t nextFrame,
        Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
