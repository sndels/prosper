#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
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
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDsLayout);

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

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_SETUP_HPP
