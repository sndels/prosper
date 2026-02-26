#ifndef PROSPER_RENDER_PARTICLES_INIT_HPP
#define PROSPER_RENDER_PARTICLES_INIT_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::particles
{

class Init
{
  public:
    Init() noexcept = default;
    ~Init() = default;

    Init(const Init &other) = delete;
    Init(Init &&other) = delete;
    Init &operator=(const Init &other) = delete;
    Init &operator=(Init &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        const scene::WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const scene::WorldDSLayouts &worldDSLayouts);

    void drawUi(const scene::Scene &scene);

    struct InputOutput
    {
        gfx::Buffer &particles;
        gfx::Buffer &particlesFreelist;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::World &world, const InputOutput &inOut,
        uint32_t nextFrame);

  private:
    bool m_initialized{false};
    ComputePass m_computePass;

    uint32_t m_sourceDrawInstanceIndex{0};
};

} // namespace render::particles

#endif // PROSPER_RENDER_PARTICLES_INIT_HPP
