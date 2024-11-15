#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldCombine
{
  public:
    DepthOfFieldCombine() noexcept = default;
    ~DepthOfFieldCombine() = default;

    DepthOfFieldCombine(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine(DepthOfFieldCombine &&other) = delete;
    DepthOfFieldCombine &operator=(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine &operator=(DepthOfFieldCombine &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

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
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
