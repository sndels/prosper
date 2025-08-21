#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP

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

class Flatten
{
  public:
    static const uint32_t sFlattenFactor = 8;

    Flatten() noexcept = default;
    ~Flatten() = default;

    Flatten(const Flatten &other) = delete;
    Flatten(Flatten &&other) = delete;
    Flatten &operator=(const Flatten &other) = delete;
    Flatten &operator=(Flatten &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Output
    {
        ImageHandle tileMinMaxCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle halfResCircleOfConfusion, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_FLATTEN_HPP
