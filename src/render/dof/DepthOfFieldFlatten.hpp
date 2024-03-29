#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldFlatten
{
  public:
    DepthOfFieldFlatten() noexcept = default;
    ~DepthOfFieldFlatten() = default;

    DepthOfFieldFlatten(const DepthOfFieldFlatten &other) = delete;
    DepthOfFieldFlatten(DepthOfFieldFlatten &&other) = delete;
    DepthOfFieldFlatten &operator=(const DepthOfFieldFlatten &other) = delete;
    DepthOfFieldFlatten &operator=(DepthOfFieldFlatten &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Output
    {
        ImageHandle tileMinMaxCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle halfResCircleOfConfusion, uint32_t nextFrame,
        Profiler *profiler);

  private:
    bool _initialized{false};
    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP
