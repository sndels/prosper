#include "Simulate.hpp"

#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/particles/Particles.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"
#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/glm.hpp>
#include <imgui.h>
#include <shader_structs/push_constants/particles/simulate.h>

using namespace glm;
using namespace wheels;

namespace render::particles
{

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/particles/simulate.comp",
        .debugName = String{alloc, "ParticlesSimulateCS"},
        .groupSize = {256, 1, 1},
    };
}

} // namespace

void Simulate::init(wheels::ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(scopeAlloc.child_scope(), shaderDefinitionCallback);

    m_initialized = true;
}

void Simulate::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void Simulate::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const InputOutput &inOut,
    float deltaTimeS, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Simulate::Particles");

    {
        const StaticArray descriptorInfos{{
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = inOut.particles.handle,
                .range = VK_WHOLE_SIZE,
            }},
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = inOut.particlesFreelist.handle,
                .range = VK_WHOLE_SIZE,
            }},
        }};
        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        const StaticArray bufferBarriers{{
            *inOut.particles.transitionBarrier(
                gfx::BufferState::ComputeShaderReadWrite),
            *inOut.particlesFreelist.transitionBarrier(
                gfx::BufferState::ComputeShaderReadWrite),
        }};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount =
                asserted_cast<uint32_t>(bufferBarriers.size()),
            .pBufferMemoryBarriers = bufferBarriers.data(),
        });

        PROFILER_GPU_SCOPE(cb, "  Simulate::Particles");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{Particles::sMaxParticleCount, 1u, 1u});

        m_computePass.record(
            cb,
            SimulatePC{
                .maxParticleCount = Particles::sMaxParticleCount,
                .deltaTimeS = deltaTimeS,
            },
            groupCount, Span{&storageSet, 1});
    }
}

} // namespace render::particles
