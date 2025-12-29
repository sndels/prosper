#include "Particles.hpp"

#include "render/RenderResources.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"

using namespace wheels;

namespace render::particles
{

void Particles::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout cameraDsLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    m_initPass.init(scopeAlloc.child_scope(), worldDSLayouts);
    m_renderPass.init(scopeAlloc.child_scope(), cameraDsLayout);

    m_initialized = true;
}

void Particles::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout cameraDsLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(m_initialized);

    m_initPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, worldDSLayouts);
    m_renderPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, cameraDsLayout);
}

void Particles::drawUi() { }

void Particles::record(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const scene::Camera &cam, const scene::World &world,
    const InputOutput &inOut, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Particles");

    {
        const Init::Output initOutput =
            m_initPass.record(scopeAlloc.child_scope(), cb, world, nextFrame);

        m_renderPass.record(
            scopeAlloc.child_scope(), cb, cam,
            Render::InputOutput{
                .inParticles = initOutput.particles,
                .inIndirectArgs = initOutput.indirectArgs,
                .inOutIllumination = inOut.illumination,
                .inOutDepth = inOut.depth,
            },
            nextFrame);

        gRenderResources.buffers->release(initOutput.particles);
        gRenderResources.buffers->release(initOutput.indirectArgs);
    }
}

} // namespace render::particles
