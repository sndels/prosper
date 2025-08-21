#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP

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

class Combine
{
  public:
    Combine() noexcept = default;
    ~Combine() = default;

    Combine(const Combine &other) = delete;
    Combine(Combine &&other) = delete;
    Combine &operator=(const Combine &other) = delete;
    Combine &operator=(Combine &&other) = delete;

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

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_COMBINE_HPP
