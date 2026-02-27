#ifndef PROSPER_RENDER_PARTICLES_HPP
#define PROSPER_RENDER_PARTICLES_HPP

#include "gfx/Resources.hpp"
#include "render/particles/Decay.hpp"
#include "render/particles/Init.hpp"
#include "render/particles/Render.hpp"
#include "render/particles/Simulate.hpp"
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
    ~Particles();

    Particles(const Particles &other) = delete;
    Particles(Particles &&other) = delete;
    Particles &operator=(const Particles &other) = delete;
    Particles &operator=(Particles &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        vk::DescriptorSetLayout cameraDsLayout,
        const scene::WorldDSLayouts &worldDSLayouts);
    void destroy();

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDsLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void drawUi(const scene::Scene &scene);

    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const scene::World &world,
        const InputOutput &inOut, float deltaTimeS, uint32_t nextFrame);

  private:
    bool m_initialized{false};

    gfx::Buffer m_particles;
    gfx::Buffer m_particlesFreelist;
    vk::DescriptorSetLayout m_particlesRWLayout;
    vk::DescriptorSetLayout m_particlesRenderLayout;
    vk::DescriptorSet m_particlesRWDS;
    vk::DescriptorSet m_particlesRenderDS;

    uint32_t m_sourceDrawInstanceIndex{0};
    bool m_resetParticles{true};

    Init m_initPass;
    Decay m_decayPass;
    Simulate m_simulatePass;
    Render m_renderPass;
};

} // namespace render::particles

#endif // PROSPER_RENDER_PARTICLES_HPP
