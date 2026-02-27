#include "Particles.hpp"

#include "gfx/Device.hpp"
#include "scene/Scene.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"
#include "utils/Ui.hpp"
#include "vulkan/vulkan.hpp"

#include <shader_structs/particles/particle.h>

using namespace wheels;
using namespace glm;

namespace render::particles
{

Particles::~Particles()
{
    gfx::gDevice.destroy(m_particles);
    gfx::gDevice.destroy(m_particlesFreelist);
}

void Particles::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout cameraDsLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    {
        ScopedScratch tmpScope = scopeAlloc.child_scope();
        Array<::particles::shader_structs::Particle> particlesInit(tmpScope);
        particlesInit.resize(sMaxParticleCount);
        m_particles = gfx::gDevice.create(gfx::BufferCreateInfo{
            .desc =
                gfx::BufferDescription{
                    .byteSize = sizeof(::particles::shader_structs::Particle) *
                                sMaxParticleCount,
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .initialData = particlesInit.data(),
            .debugName = "Particles"});
    }

    {
        ScopedScratch tmpScope = scopeAlloc.child_scope();
        Array<int32_t> freelistInit(tmpScope, 1 + sMaxParticleCount);
        const int32_t indexCount = asserted_cast<int32_t>(sMaxParticleCount);
        freelistInit.emplace_back(indexCount);
        for (int32_t i = 0; i < indexCount; ++i)
            freelistInit.emplace_back(i);
        m_particlesFreelist = gfx::gDevice.create(gfx::BufferCreateInfo{
            .desc =
                gfx::BufferDescription{
                    .byteSize = freelistInit.size() * sizeof(int32_t),
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .initialData = freelistInit.data(),
            .debugName = "ParticlesFreelist"});
    }

    m_decayPass.init(scopeAlloc.child_scope());
    m_initPass.init(scopeAlloc.child_scope(), worldDSLayouts);
    m_simulatePass.init(scopeAlloc.child_scope());
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

    m_decayPass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_initPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, worldDSLayouts);
    m_simulatePass.recompileShaders(scopeAlloc.child_scope(), changedFiles);
    m_renderPass.recompileShaders(
        scopeAlloc.child_scope(), changedFiles, cameraDsLayout);
}

void Particles::drawUi(const scene::Scene &scene)
{
    WHEELS_ASSERT(m_initialized);

    if (utils::sliderU32(
            "Source mesh", &m_sourceDrawInstanceIndex, 0u,
            scene.drawInstanceCount - 1))
    {
        m_resetParticles = true;
    }
    m_resetParticles |= ImGui::Button("Reset particles");
}

void Particles::record(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const scene::Camera &cam, const scene::World &world,
    const InputOutput &inOut, float deltaTimeS, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Particles");

    {
        m_decayPass.record(
            scopeAlloc.child_scope(), cb,
            Decay::InputOutput{
                .particles = m_particles,
                .particlesFreelist = m_particlesFreelist,
            },
            m_resetParticles, nextFrame);

        if (m_resetParticles)
        {
            const bool initRecorded = m_initPass.record(
                scopeAlloc.child_scope(), cb, world, m_sourceDrawInstanceIndex,
                Init::InputOutput{
                    .particles = m_particles,
                    .particlesFreelist = m_particlesFreelist,
                },
                nextFrame);
            if (initRecorded)
                m_resetParticles = false;
        }

        m_simulatePass.record(
            scopeAlloc.child_scope(), cb,
            Simulate::InputOutput{
                .particles = m_particles,
                .particlesFreelist = m_particlesFreelist,
            },
            deltaTimeS, nextFrame);

        m_renderPass.record(
            scopeAlloc.child_scope(), cb, cam,
            Render::InputOutput{
                .inParticles = m_particles,
                .inOutIllumination = inOut.illumination,
                .inOutDepth = inOut.depth,
            },
            nextFrame);
    }
}

} // namespace render::particles
