#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "../../gfx/Fwd.hpp"
#include "../../gfx/Resources.hpp"
#include "../../utils/Fwd.hpp"
#include "../../utils/Utils.hpp"
#include "../ComputePass.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

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
    Buffer m_atomicCounter;
    bool m_counterNotCleared{true};
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_REDUCE_HPP
