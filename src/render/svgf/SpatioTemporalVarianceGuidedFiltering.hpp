#ifndef PROSPER_RENDER_SVGF_SPATIO_TEMPORAL_VARIANCE_GUIDED_FILTERING_HPP
#define PROSPER_RENDER_SVGF_SPATIO_TEMPORAL_VARIANCE_GUIDED_FILTERING_HPP

#include "render/Fwd.hpp"
#include "render/svgf/Accumulate.hpp"
#include "scene/Fwd.hpp"

#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::svgf
{

using Input = Accumulate::Input;

class SpatioTemporalVarianceGuidedFiltering
{
  public:
    SpatioTemporalVarianceGuidedFiltering() noexcept = default;
    ~SpatioTemporalVarianceGuidedFiltering() = default;

    SpatioTemporalVarianceGuidedFiltering(
        const SpatioTemporalVarianceGuidedFiltering &other) = delete;
    SpatioTemporalVarianceGuidedFiltering(
        SpatioTemporalVarianceGuidedFiltering &&other) = delete;
    SpatioTemporalVarianceGuidedFiltering &operator=(
        const SpatioTemporalVarianceGuidedFiltering &other) = delete;
    SpatioTemporalVarianceGuidedFiltering &operator=(
        SpatioTemporalVarianceGuidedFiltering &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout);

    void drawUi();

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const Input &input, uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};

    bool m_ignoreHistory{true};

    Accumulate m_accumulate;
};

} // namespace render::svgf

#endif // PROSPER_RENDER_SVGF_SPATIO_TEMPORAL_VARIANCE_GUIDED_FILTERING_HPP
