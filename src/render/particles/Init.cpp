#include "Init.hpp"

#include "gfx/Resources.hpp"
#include "render/ComputePass.hpp"
#include "render/particles/Particles.hpp"
#include "scene/Mesh.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"
#include "utils/Ui.hpp"
#include "utils/Utils.hpp"
#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/glm.hpp>
#include <imgui.h>
#include <shader_structs/push_constants/particles/init.h>

using namespace glm;
using namespace wheels;

namespace render::particles
{

namespace
{

enum InitBindingSet : uint8_t
{
    GeometryBindingSet,
    SceneInstancesBindingSet,
    StorageBindingSet,
    BindingSetCount
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 96;
    String defines{alloc, len};
    appendDefineStr(defines, "GEOMETRY_SET", GeometryBindingSet);
    appendDefineStr(defines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/particles/init.comp",
        .debugName = String{alloc, "ParticlesInitCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = {256, 1, 1},
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const scene::WorldDSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[GeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[SceneInstancesBindingSet] = worldDSLayouts.sceneInstances;
    return setLayouts;
}

} // namespace

void Init::init(
    wheels::ScopedScratch scopeAlloc,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        scopeAlloc.child_scope(), shaderDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = externalDsLayouts(worldDSLayouts),
        });

    m_initialized = true;
}

void Init::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(m_initialized);

    m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, shaderDefinitionCallback,
        externalDsLayouts(worldDSLayouts));
}

void Init::drawUi(const scene::Scene &scene)
{
    WHEELS_ASSERT(m_initialized);

    utils::sliderU32(
        "Source mesh", &m_sourceDrawInstanceIndex, 0u,
        scene.drawInstanceCount - 1);
}

void Init::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::World &world,
    const InputOutput &inOut, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Init::Particles");

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
                gfx::BufferState::ComputeShaderWrite),
            *inOut.particlesFreelist.transitionBarrier(
                gfx::BufferState::ComputeShaderReadWrite),
        }};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount =
                asserted_cast<uint32_t>(bufferBarriers.size()),
            .pBufferMemoryBarriers = bufferBarriers.data(),
        });

        const scene::Scene &currentScene = world.currentScene();
        const scene::WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[GeometryBindingSet] =
            world.descriptorSets().geometry[nextFrame];
        descriptorSets[SceneInstancesBindingSet] =
            currentScene.sceneInstancesDescriptorSet;
        descriptorSets[StorageBindingSet] = storageSet;

        const StaticArray dynamicOffsets{{
            worldByteOffsets.modelInstanceTransforms,
            worldByteOffsets.previousModelInstanceTransforms,
            worldByteOffsets.modelInstanceScales,
        }};

        PROFILER_GPU_SCOPE(cb, "  Init::Particles");

        const scene::shader_structs::DrawInstance &drawInstance =
            currentScene.drawInstances[m_sourceDrawInstanceIndex];
        const scene::MeshInfo &meshInfo =
            world.meshInfos()[drawInstance.meshIndex];
        // Mesh might not have been loaded in yet
        if (meshInfo.indexCount > 0)
        {
            const uint32_t vertexCount = meshInfo.vertexCount;
            const uvec3 groupCount =
                m_computePass.groupCount(uvec3{vertexCount, 1u, 1u});

            m_computePass.record(
                cb,
                InitPC{
                    .drawInstanceIndex = m_sourceDrawInstanceIndex,
                    .vertexCount = meshInfo.vertexCount,
                    .maxParticleCount = Particles::sMaxParticleCount,
                },
                groupCount, descriptorSets,
                ComputePassOptionalRecordArgs{
                    .dynamicOffsets = dynamicOffsets,
                });
        }
    }
}

} // namespace render::particles
