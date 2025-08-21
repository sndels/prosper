#ifndef PROSPER_RENDER_BLOOM_REDUCE_HPP
#define PROSPER_RENDER_BLOOM_REDUCE_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::bloom
{

class Reduce
{
  public:
    Reduce() noexcept = default;
    ~Reduce();

    Reduce(const Reduce &other) = delete;
    Reduce(Reduce &&other) = delete;
    Reduce &operator=(const Reduce &other) = delete;
    Reduce &operator=(Reduce &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        ImageHandle inOutHighlightMips, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
    gfx::Buffer m_atomicCounter;
    bool m_counterNotCleared{true};
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_REDUCE_HPP
