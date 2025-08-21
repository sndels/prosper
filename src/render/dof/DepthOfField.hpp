#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_HPP

#include "render/dof/Combine.hpp"
#include "render/dof/Dilate.hpp"
#include "render/dof/Filter.hpp"
#include "render/dof/Flatten.hpp"
#include "render/dof/Gather.hpp"
#include "render/dof/Reduce.hpp"
#include "render/dof/Setup.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::dof
{

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

using Input = Setup::Input;
using Output = Combine::Output;

class DepthOfField
{
  public:
    // Foreground can have an (almost?) infinitely larger bokeh so let's clamp
    // to a smaller but still plausible looking factor
    static constexpr float sMaxFgCoCFactor = 2.f;

    DepthOfField() noexcept = default;
    ~DepthOfField() = default;

    DepthOfField(const DepthOfField &other) = delete;
    DepthOfField(DepthOfField &&other) = delete;
    DepthOfField &operator=(const DepthOfField &other) = delete;
    DepthOfField &operator=(DepthOfField &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        vk::DescriptorSetLayout cameraDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDsLayout);

    void startFrame();

    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};

    Setup m_setupPass;
    Reduce m_reducePass;
    Flatten m_flattenPass;
    Dilate m_dilatePass;
    Gather m_gatherPass;
    Filter m_filterPass;
    Combine m_combinePass;
};

} // namespace render::dof

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_HPP
