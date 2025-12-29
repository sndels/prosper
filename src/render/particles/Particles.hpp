#ifndef PROSPER_RENDER_PARTICLES_HPP
#define PROSPER_RENDER_PARTICLES_HPP

#include "render/particles/Init.hpp"
#include "render/particles/Render.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::particles
{

class Particles
{
  public:
    struct InputOutput
    {
        ImageHandle illumination;
        ImageHandle depth;
    };

    static constexpr uint32_t sMaxParticleCount{500'000};

    Particles() noexcept = default;
    ~Particles() = default;

    Particles(const Particles &other) = delete;
    Particles(Particles &&other) = delete;
    Particles &operator=(const Particles &other) = delete;
    Particles &operator=(Particles &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        vk::DescriptorSetLayout cameraDsLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDsLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void drawUi();

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const scene::World &world,
        const InputOutput &inOut, uint32_t nextFrame);

  private:
    bool m_initialized{false};

    Init m_initPass;
    Render m_renderPass;
};

} // namespace render::particles

#endif // PROSPER_RENDER_PARTICLES_HPP
