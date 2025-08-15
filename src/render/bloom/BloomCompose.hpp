#ifndef PROSPER_RENDER_BLOOM_COMPOSE_HPP
#define PROSPER_RENDER_BLOOM_COMPOSE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/BloomResolutionScale.hpp"
#include "render/bloom/BloomTechnique.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

namespace render::bloom
{

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

    void drawUi(BloomTechnique technique);

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
        BloomTechnique technique, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    bool m_biquadraticSampling{true};
    glm::vec3 m_blendFactors{.9f, .04f, .04f};
    ComputePass m_computePass;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_COMPOSE_HPP
