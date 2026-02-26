#include "Decay.hpp"

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
#include <shader_structs/push_constants/particles/decay.h>

using namespace glm;
using namespace wheels;

namespace render::particles
{

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    return ComputePass::Shader{
        .relPath = "shader/particles/decay.comp",
        .debugName = String{alloc, "ParticlesDecayCS"},
        .groupSize = {256, 1, 1},
    };
}

} // namespace

void Decay::init(wheels::ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(scopeAlloc.child_scope(), shaderDefinitionCallback);

    m_initialized = true;
}

void Decay::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback);
}

void Decay::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const InputOutput &inOut,
    uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Decay::Particles");

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

        PROFILER_GPU_SCOPE(cb, "  Decay::Particles");

        const uvec3 groupCount = m_computePass.groupCount(
            uvec3{Particles::sMaxParticleCount, 1u, 1u});

        m_computePass.record(
            cb,
            DecayPC{
                // TODO: This should be a constant since it's compile time in
                // C++ too
                .maxParticleCount = Particles::sMaxParticleCount,
            },
            groupCount, Span{&storageSet, 1});
    }
}

} // namespace render::particles
