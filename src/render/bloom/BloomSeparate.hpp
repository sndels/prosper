#ifndef PROSPER_RENDER_BLOOM_SEPARATE_HPP
#define PROSPER_RENDER_BLOOM_SEPARATE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

class BloomSeparate
{
  public:
    BloomSeparate() noexcept = default;
    ~BloomSeparate() = default;

    BloomSeparate(const BloomSeparate &other) = delete;
    BloomSeparate(BloomSeparate &&other) = delete;
    BloomSeparate &operator=(const BloomSeparate &other) = delete;
    BloomSeparate &operator=(BloomSeparate &&other) = delete;

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
        const Input &input, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    float m_threshold{1.f};
    ComputePass m_computePass;
};

#endif // PROSPER_RENDER_BLOOM_SEPARATE_HPP
