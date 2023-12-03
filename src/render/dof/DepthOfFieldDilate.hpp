#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP

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

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Output
    {
        ImageHandle dilatedTileMinMaxCoC;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle tileMinMaxCoC, uint32_t nextFrame, Profiler *profiler);

    RenderResources *_resources{nullptr};
    ComputePass _computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
