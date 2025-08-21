#include "InitialReservoirs.hpp"
#include "render/GBufferRenderer.hpp"
#include "render/RenderResources.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "scene/Light.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <shader_structs/push_constants/restir_di/initial_reservoirs.h>

using namespace glm;
using namespace wheels;

namespace render::rtdi
{

namespace
{

constexpr uint32_t sFramePeriod = 4096;

enum BindingSet : uint8_t
{
    LightsBindingSet,
    CameraBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

ComputePass::Shader shaderDefinitionCallback(
    Allocator &alloc, const scene::WorldDSLayouts &worldDSLayouts)
{
    const size_t len = 768;
    String defines{alloc, len};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    scene::PointLights::appendShaderDefines(defines);
    scene::SpotLights::appendShaderDefines(defines);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/restir_di/initial_reservoirs.comp",
        .debugName = String{alloc, "RtDiInitialReservoirsCS"},
        .defines = WHEELS_MOV(defines),
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const InitialReservoirs::InputDSLayouts &dsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    return setLayouts;
}

} // namespace

void InitialReservoirs::init(
    ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc), [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = externalDsLayouts(dsLayouts),
        });

    m_initialized = true;
}

bool InitialReservoirs::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(m_initialized);

    return m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        externalDsLayouts(dsLayouts));
}

InitialReservoirs::Output InitialReservoirs::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::World &world,
    const scene::Camera &cam, const GBufferRendererOutput &gbuffer,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  InitialReservoirs");

    m_frameIndex = (m_frameIndex + 1) % sFramePeriod;

    Output ret;
    {
        const vk::Extent2D renderExtent = getExtent2D(gbuffer.albedoRoughness);

        ret.reservoirs = gRenderResources.images->create(
            gfx::ImageDescription{
                .format = vk::Format::eR32G32Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eStorage |
                              vk::ImageUsageFlagBits::eSampled,
            },
            "InitialReservoirs");

        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(gbuffer.albedoRoughness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(gbuffer.normalMetalness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(gbuffer.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.reservoirs).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {gbuffer.albedoRoughness,
                     gfx::ImageState::ComputeShaderRead},
                    {gbuffer.normalMetalness,
                     gfx::ImageState::ComputeShaderRead},
                    {gbuffer.depth, gfx::ImageState::ComputeShaderRead},
                    {ret.reservoirs, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  InitialReservoirs");

        const scene::WorldDescriptorSets &worldDSes = world.descriptorSets();
        const scene::WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] = worldDSes.lights;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[StorageBindingSet] = storageSet;

        const StaticArray dynamicOffsets{{
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
            cam.bufferOffset(),
        }};

        const uvec3 groupCount = m_computePass.groupCount(
            glm::uvec3{renderExtent.width, renderExtent.height, 1u});

        const InitialReservoirsPC pcBlock{
            .frameIndex = m_frameIndex,
        };

        m_computePass.record(
            cb, pcBlock, groupCount, descriptorSets,
            ComputePassOptionalRecordArgs{
                .dynamicOffsets = dynamicOffsets,
            });
    }

    return ret;
}

} // namespace render::rtdi
