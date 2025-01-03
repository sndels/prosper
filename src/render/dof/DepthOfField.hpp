#ifndef PROSPER_RENDER_DEPTH_OF_FIELD_HPP
#define PROSPER_RENDER_DEPTH_OF_FIELD_HPP

#include "render/dof/DepthOfFieldCombine.hpp"
#include "render/dof/DepthOfFieldDilate.hpp"
#include "render/dof/DepthOfFieldFilter.hpp"
#include "render/dof/DepthOfFieldFlatten.hpp"
#include "render/dof/DepthOfFieldGather.hpp"
#include "render/dof/DepthOfFieldReduce.hpp"
#include "render/dof/DepthOfFieldSetup.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

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

    using Input = DepthOfFieldSetup::Input;
    using Output = DepthOfFieldCombine::Output;
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};

    DepthOfFieldSetup m_setupPass;
    DepthOfFieldReduce m_reducePass;
    DepthOfFieldFlatten m_flattenPass;
    DepthOfFieldDilate m_dilatePass;
    DepthOfFieldGather m_gatherPass;
    DepthOfFieldFilter m_filterPass;
    DepthOfFieldCombine m_combinePass;
};

#endif // PROSPER_RENDER_DEPTH_OF_FIELD_HPP
