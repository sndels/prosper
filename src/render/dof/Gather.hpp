#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::dof
{

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class Gather
{
  public:
    enum GatherType : uint8_t
    {
        GatherType_Foreground,
        GatherType_Background,
        GatherType_Count,
    };

    Gather() noexcept = default;
    ~Gather() = default;

    Gather(const Gather &other) = delete;
    Gather(Gather &&other) = delete;
    Gather &operator=(const Gather &other) = delete;
    Gather &operator=(Gather &&other) = delete;

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

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_GATHER_HPP
