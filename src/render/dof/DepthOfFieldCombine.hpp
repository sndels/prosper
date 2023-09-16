#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Device.hpp"
#include "../../scene/Camera.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../RenderResources.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldCombine
{
  public:
    DepthOfFieldCombine(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldCombine() = default;

    DepthOfFieldCombine(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine(DepthOfFieldCombine &&other) = delete;
    DepthOfFieldCombine &operator=(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine &operator=(DepthOfFieldCombine &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Input
    {
        ImageHandle halfResFgBokehWeight;
        ImageHandle halfResBgBokehWeight;
        ImageHandle halfResCircleOfConfusion;
        ImageHandle illumination;
    };
    struct Output
    {
        ImageHandle combinedIlluminationDoF;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Input &input, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
