#ifndef PROSPER_DEPTH_OF_FIELD_COMBINE_HPP
#define PROSPER_DEPTH_OF_FIELD_COMBINE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "ComputePass.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

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

#endif // PROSPER_DEPTH_OF_FIELD_COMBINE_HPP
