#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldDilate
{
  public:
    DepthOfFieldDilate() noexcept = default;
    ~DepthOfFieldDilate() = default;

    DepthOfFieldDilate(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate(DepthOfFieldDilate &&other) = delete;
    DepthOfFieldDilate &operator=(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate &operator=(DepthOfFieldDilate &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Output
    {
        ImageHandle dilatedTileMinMaxCoC;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle tileMinMaxCoC, const Camera &cam, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_DILATE_HPP
