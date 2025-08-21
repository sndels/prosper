#ifndef PROSPER_RENDER_BLOOM_COMPOSE_HPP
#define PROSPER_RENDER_BLOOM_COMPOSE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/bloom/ResolutionScale.hpp"
#include "render/bloom/Technique.hpp"

#include <glm/glm.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>

namespace render::bloom
{

class Compose
{
  public:
    Compose() noexcept = default;
    ~Compose() = default;

    Compose(const Compose &other) = delete;
    Compose(Compose &&other) = delete;
    Compose &operator=(const Compose &other) = delete;
    Compose &operator=(Compose &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void drawUi(Technique technique);

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
        const Input &input, ResolutionScale resolutionScale,
        Technique technique, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    bool m_biquadraticSampling{true};
    glm::vec3 m_blendFactors{.9f, .04f, .04f};
    ComputePass m_computePass;
};

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_COMPOSE_HPP
