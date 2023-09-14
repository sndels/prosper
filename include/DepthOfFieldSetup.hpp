#ifndef PROSPER_DEPTH_OF_FIELD_SETUP_HPP
#define PROSPER_DEPTH_OF_FIELD_SETUP_HPP

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

class DepthOfFieldSetup
{
  public:
    DepthOfFieldSetup(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDsLayout);

    ~DepthOfFieldSetup() = default;

    DepthOfFieldSetup(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup(DepthOfFieldSetup &&other) = delete;
    DepthOfFieldSetup &operator=(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup &operator=(DepthOfFieldSetup &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout);

    struct Input
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    struct Output
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Camera &cam, const Input &input,
        uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_DEPTH_OF_FIELD_SETUP_HPP
