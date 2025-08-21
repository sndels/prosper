#ifndef PROSPER_RENDER_BLOOM_SEPARATE_HPP
#define PROSPER_RENDER_BLOOM_SEPARATE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/ResolutionScale.hpp"
#include "render/bloom/Technique.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

namespace render::bloom
{

class Separate
{
  public:
    Separate() noexcept = default;
    ~Separate() = default;

    Separate(const Separate &other) = delete;
    Separate(Separate &&other) = delete;
    Separate &operator=(const Separate &other) = delete;
    Separate &operator=(Separate &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void drawUi();

    struct Input
    {
        ImageHandle illumination;
    };
    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, ResolutionScale resolutionScale,
        Technique technique, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    float m_threshold{1.f};
    ComputePass m_computePass;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_SEPARATE_HPP
