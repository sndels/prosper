#ifndef PROSPER_RENDER_HIERARCHICAL_DEPTH_DOWNSAMLPER_HPP
#define PROSPER_RENDER_HIERARCHICAL_DEPTH_DOWNSAMLPER_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/span.hpp>

namespace render
{

class HierarchicalDepthDownsampler
{
  public:
    HierarchicalDepthDownsampler() noexcept = default;
    ~HierarchicalDepthDownsampler();

    HierarchicalDepthDownsampler(const HierarchicalDepthDownsampler &other) =
        delete;
    HierarchicalDepthDownsampler(HierarchicalDepthDownsampler &&other) = delete;
    HierarchicalDepthDownsampler &operator=(
        const HierarchicalDepthDownsampler &other) = delete;
    HierarchicalDepthDownsampler &operator=(
        HierarchicalDepthDownsampler &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void startFrame();

    // Downsamples a depth pyramid, keeping it non-linear to match the input.
    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inNonLinearDepth, uint32_t nextFrame,
        wheels::StrSpan debugPrefix);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
    gfx::Buffer m_atomicCounter;
    bool m_counterNotCleared{true};
};

} // namespace render

#endif // PROSPER_RENDER_HIERARCHICAL_DEPTH_DOWNSAMPLER_HPP
