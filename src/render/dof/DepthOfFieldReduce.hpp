#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::dof
{

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldReduce
{
  public:
    DepthOfFieldReduce() noexcept = default;
    ~DepthOfFieldReduce();

    DepthOfFieldReduce(const DepthOfFieldReduce &other) = delete;
    DepthOfFieldReduce(DepthOfFieldReduce &&other) = delete;
    DepthOfFieldReduce &operator=(const DepthOfFieldReduce &other) = delete;
    DepthOfFieldReduce &operator=(DepthOfFieldReduce &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const ImageHandle &inOutIlluminationMips, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
    gfx::Buffer m_atomicCounter;
    bool m_counterNotCleared{true};
};

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP
