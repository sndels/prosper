#ifndef PROSPER_RENDER_PARTICLES_SIMULATE_HPP
#define PROSPER_RENDER_PARTICLES_SIMULATE_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::particles
{

class Simulate
{
  public:
    Simulate() noexcept = default;
    ~Simulate() = default;

    Simulate(const Simulate &other) = delete;
    Simulate(Simulate &&other) = delete;
    Simulate &operator=(const Simulate &other) = delete;
    Simulate &operator=(Simulate &&other) = delete;

    void init(wheels::ScopedScratch scopeAlloc);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles);

    struct InputOutput
    {
        gfx::Buffer &particles;
        gfx::Buffer &particlesFreelist;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const InputOutput &inOut, float deltaTimeS, uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;
};

} // namespace render::particles

#endif // PROSPER_RENDER_PARTICLES_SIMULATE_HPP
