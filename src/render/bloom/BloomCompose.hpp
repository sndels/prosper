#ifndef PROSPER_RENDER_BLOOM_COMPOSE_HPP
#define PROSPER_RENDER_BLOOM_COMPOSE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/BloomResolutionScale.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

class BloomCompose
{
  public:
    BloomCompose() noexcept = default;
    ~BloomCompose() = default;

    BloomCompose(const BloomCompose &other) = delete;
    BloomCompose(BloomCompose &&other) = delete;
    BloomCompose &operator=(const BloomCompose &other) = delete;
    BloomCompose &operator=(BloomCompose &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct Input
    {
        ImageHandle illumination;
        ImageHandle bloomHighlights;
    };
    [[nodiscard]] ImageHandle record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Input &input, BloomResolutionScale resolutionScale,
        uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_COMPOSE_HPP
