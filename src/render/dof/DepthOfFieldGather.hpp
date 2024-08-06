#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldGather
{
  public:
    enum GatherType : uint32_t
    {
        GatherType_Foreground,
        GatherType_Background,
        GatherType_Count,
    };

    DepthOfFieldGather() noexcept = default;
    ~DepthOfFieldGather() = default;

    DepthOfFieldGather(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather(DepthOfFieldGather &&other) = delete;
    DepthOfFieldGather &operator=(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather &operator=(DepthOfFieldGather &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Input
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCoC;
        ImageHandle dilatedTileMinMaxCoC;
    };
    struct Output
    {
        ImageHandle halfResBokehColorWeight;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, GatherType gatherType, uint32_t nextFrame);

  private:
    bool m_initialized{false};

    ComputePass m_backgroundPass;
    ComputePass m_foregroundPass;

    uint32_t m_frameIndex{0};
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
