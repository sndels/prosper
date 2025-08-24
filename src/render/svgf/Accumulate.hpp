#ifndef PROSPER_RENDER_SVGF_ACCUMULATE_HPP
#define PROSPER_RENDER_SVGF_ACCUMULATE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/GBuffer.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::svgf
{

class Accumulate
{
  public:
    Accumulate() noexcept = default;
    ~Accumulate() = default;

    Accumulate(const Accumulate &other) = delete;
    Accumulate(Accumulate &&other) = delete;
    Accumulate &operator=(const Accumulate &other) = delete;
    Accumulate &operator=(Accumulate &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDsLayout);

    // Returns true if recompile happened
    bool recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout);

    struct Input
    {
        GBuffer gbuffer;
        GBuffer previous_gbuffer;
        ImageHandle color;
    };
    struct Output
    {
        ImageHandle color;
        ImageHandle moments;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const Input &input, bool ignoreHistory,
        uint32_t nextFrame);
    void releasePreserved();

  private:
    bool m_initialized{false};

    ComputePass m_computePass;
    ImageHandle m_previousIntegratedMoments;
};

} // namespace render::svgf

#endif // PROSPER_RENDER_SVGF_ACCUMULATE_HPP
