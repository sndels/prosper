#include "Particles.hpp"

#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"
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
        Array<::particles::shader_structs::Particle> particlesInit(
            tmpScope, sMaxParticleCount);
        for (uint32_t i = 0; i < sMaxParticleCount; ++i)
            particlesInit.emplace_back(::particles::shader_structs::Particle{
                .position_lifetime = vec4{-1.f},
            });
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
        m_initPass.record(
            scopeAlloc.child_scope(), cb, world,
            Init::InputOutput{
                .particles = m_particles,
                .particlesFreelist = m_particlesFreelist,
            },
            nextFrame);

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
