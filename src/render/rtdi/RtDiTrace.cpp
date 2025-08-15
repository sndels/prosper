#include "RtDiTrace.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "gfx/VkUtils.hpp"
#include "render/GBufferRenderer.hpp"
#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/restir_di/trace.h>

using namespace wheels;

namespace render::rtdi
{

// Based on RT Gems II chapter 16

namespace
{

constexpr uint32_t sFramePeriod = 4096;

enum BindingSet : uint8_t
{
    CameraBindingSet,
    RTBindingSet,
    StorageBindingSet,
    MaterialDatasBindingSet,
    MaterialTexturesBindingSet,
    GeometryBindingSet,
    SkyboxBindingSet,
    SceneInstancesBindingSet,
    LightsBindingSet,
    BindingSetCount,
};

constexpr vk::ShaderStageFlags sVkShaderStageFlagsAllRt =
    vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eAnyHitKHR |
    vk::ShaderStageFlagBits::eClosestHitKHR |
    vk::ShaderStageFlagBits::eMissKHR |
    vk::ShaderStageFlagBits::eIntersectionKHR;

enum class StageIndex : uint8_t
{
    RayGen,
    ClosestHit,
    AnyHit,
    Miss,
};

enum class GroupIndex : uint8_t
{
    RayGen,
    Hit,
    Miss,
};

struct TracePCFlags
{
    bool skipHistory{false};
    bool accumulate{false};
};

inline uint32_t pcFlags(TracePCFlags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.skipHistory;
    ret |= (uint32_t)flags.accumulate << 1;

    return ret;
}

} // namespace

RtDiTrace::~RtDiTrace()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    destroyPipeline();

    gfx::gDevice.logical().destroy(m_descriptorSetLayout);

    gfx::gDevice.destroy(m_shaderBindingTable);
    destroyShaders();
}

void RtDiTrace::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating RtDiTrace");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RtDiTrace shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope());
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());

    m_initialized = true;
}

void RtDiTrace::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_raygenReflection.has_value());
    WHEELS_ASSERT(m_rayMissReflection.has_value());
    WHEELS_ASSERT(m_closestHitReflection.has_value());
    WHEELS_ASSERT(m_anyHitReflection.has_value());
    if (!m_raygenReflection->affected(changedFiles) &&
        !m_rayMissReflection->affected(changedFiles) &&
        !m_closestHitReflection->affected(changedFiles) &&
        !m_anyHitReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
        m_accumulationDirty = true;
    }
}

RtDiTrace::Output RtDiTrace::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, scene::World &world,
    const scene::Camera &cam, const Input &input, bool resetAccumulation,
    scene::DrawType drawType, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Trace");

    m_frameIndex = (m_frameIndex + 1) % sFramePeriod;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getExtent2D(input.gbuffer.albedoRoughness);

        const gfx::ImageDescription accumulateImageDescription =
            gfx::ImageDescription{
                .format = vk::Format::eR32G32B32A32Sfloat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags = vk::ImageUsageFlagBits::eStorage |
                              vk::ImageUsageFlagBits::eTransferSrc |
                              vk::ImageUsageFlagBits::eSampled,
            };
        // TODO:
        // This could be a 'normal' lower bitdepth illumination target when
        // accumulation is skipped. However, glsl needs explicit format for the
        // uniform.
        ImageHandle illumination = gRenderResources.images->create(
            accumulateImageDescription, "RtDiTrace32bit");

        vk::Extent3D previousExtent;
        if (gRenderResources.images->isValidHandle(m_previousIllumination))
            previousExtent =
                gRenderResources.images->resource(m_previousIllumination)
                    .extent;

        if (resetAccumulation || renderExtent.width != previousExtent.width ||
            renderExtent.height != previousExtent.height)
        {
            if (gRenderResources.images->isValidHandle(m_previousIllumination))
                gRenderResources.images->release(m_previousIllumination);

            // Create dummy texture that won't be read from to satisfy binds
            m_previousIllumination = gRenderResources.images->create(
                accumulateImageDescription, "previousRtDiTrace");
            m_accumulationDirty = true;
        }
        else // We clear debug names each frame
            gRenderResources.images->appendDebugName(
                m_previousIllumination, "previousRtDiTrace");

        updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, input, illumination);

        world.currentTLAS().buffer.transition(
            cb, gfx::BufferState::RayTracingAccelerationStructureRead);

        transition(
            scopeAlloc.child_scope(), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 6>{{
                    {input.gbuffer.albedoRoughness,
                     gfx::ImageState::RayTracingRead},
                    {input.gbuffer.normalMetalness,
                     gfx::ImageState::RayTracingRead},
                    {input.gbuffer.depth, gfx::ImageState::RayTracingRead},
                    {input.reservoirs, gfx::ImageState::RayTracingRead},
                    {m_previousIllumination, gfx::ImageState::RayTracingRead},
                    {illumination, gfx::ImageState::RayTracingReadWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Trace");

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_pipeline);

        const auto &scene = world.currentScene();
        const scene::WorldDescriptorSets &worldDSes = world.descriptorSets();
        const scene::WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[RTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[StorageBindingSet] = m_descriptorSets[nextFrame];
        descriptorSets[MaterialDatasBindingSet] =
            worldDSes.materialDatas[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
        descriptorSets[GeometryBindingSet] = worldDSes.geometry[nextFrame];
        descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
        descriptorSets[SceneInstancesBindingSet] =
            scene.sceneInstancesDescriptorSet;
        descriptorSets[LightsBindingSet] = worldDSes.lights;

        const StaticArray dynamicOffsets{{
            cam.bufferOffset(),
            worldByteOffsets.globalMaterialConstants,
            worldByteOffsets.modelInstanceTransforms,
            worldByteOffsets.previousModelInstanceTransforms,
            worldByteOffsets.modelInstanceScales,
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
        }};

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR, m_pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        const TracePC pcBlock{
            .drawType = static_cast<uint32_t>(drawType),
            .frameIndex = m_frameIndex,
            .flags = pcFlags(TracePCFlags{
                .skipHistory = cam.changedThisFrame() || resetAccumulation ||
                               m_accumulationDirty,
                .accumulate = m_accumulate,
            }),
        };
        cb.pushConstants(
            m_pipelineLayout, sVkShaderStageFlagsAllRt, 0, sizeof(pcBlock),
            &pcBlock);

        WHEELS_ASSERT(m_shaderBindingTable.deviceAddress != 0);
        const vk::DeviceAddress sbtAddr = m_shaderBindingTable.deviceAddress;

        const vk::StridedDeviceAddressRegionKHR rayGenRegion{
            .deviceAddress =
                sbtAddr +
                (m_sbtGroupSize * static_cast<uint32_t>(GroupIndex::RayGen)),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR missRegion{
            .deviceAddress = sbtAddr + (m_sbtGroupSize * static_cast<uint32_t>(
                                                             GroupIndex::Miss)),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR hitRegion{
            .deviceAddress = sbtAddr + (m_sbtGroupSize *
                                        static_cast<uint32_t>(GroupIndex::Hit)),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR callableRegion;

        cb.traceRaysKHR(
            &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
            renderExtent.width, renderExtent.height, 1);

        gRenderResources.images->release(m_previousIllumination);
        m_previousIllumination = illumination;
        gRenderResources.images->preserve(m_previousIllumination);

        // Further passes expect 16bit illumination with pipelines created with
        // the attachment format
        {
            ret.illumination = createIllumination(renderExtent, "RtDiTrace");
            transition(
                WHEELS_MOV(scopeAlloc), cb,
                Transitions{
                    .images = StaticArray<ImageTransition, 2>{{
                        {illumination, gfx::ImageState::TransferSrc},
                        {ret.illumination, gfx::ImageState::TransferDst},
                    }},
                });
            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};
            const std::array offsets{
                vk::Offset3D{.x = 0, .y = 0, .z = 0},
                vk::Offset3D{
                    .x = asserted_cast<int32_t>(renderExtent.width),
                    .y = asserted_cast<int32_t>(renderExtent.height),
                    .z = 1,
                },
            };
            const auto blit = vk::ImageBlit{
                .srcSubresource = layers,
                .srcOffsets = offsets,
                .dstSubresource = layers,
                .dstOffsets = offsets,
            };
            cb.blitImage(
                gRenderResources.images->nativeHandle(illumination),
                vk::ImageLayout::eTransferSrcOptimal,
                gRenderResources.images->nativeHandle(ret.illumination),
                vk::ImageLayout::eTransferDstOptimal, 1, &blit,
                vk::Filter::eLinear);
        }
    }

    m_accumulationDirty = false;

    return ret;
}

void RtDiTrace::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    if (gRenderResources.images->isValidHandle(m_previousIllumination))
        gRenderResources.images->release(m_previousIllumination);
}

void RtDiTrace::destroyShaders()
{
    for (auto const &stage : m_shaderStages)
        gfx::gDevice.logical().destroyShaderModule(stage.module);
}

void RtDiTrace::destroyPipeline()
{
    gfx::gDevice.logical().destroy(m_pipeline);
    gfx::gDevice.logical().destroy(m_pipelineLayout);
}

bool RtDiTrace::compileShaders(
    ScopedScratch scopeAlloc, const scene::WorldDSLayouts &worldDSLayouts)
{
    const size_t raygenDefsLen = 768;
    String raygenDefines{scopeAlloc, raygenDefsLen};
    appendDefineStr(raygenDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(raygenDefines, "RAY_TRACING_SET", RTBindingSet);
    appendDefineStr(raygenDefines, "STORAGE_SET", StorageBindingSet);
    appendEnumVariantsAsDefines(
        raygenDefines, "DrawType",
        Span{scene::sDrawTypeNames.data(), scene::sDrawTypeNames.size()});
    appendDefineStr(
        raygenDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        raygenDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        raygenDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(raygenDefines, "GEOMETRY_SET", GeometryBindingSet);
    appendDefineStr(raygenDefines, "SKYBOX_SET", SkyboxBindingSet);
    appendDefineStr(
        raygenDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(raygenDefines, "LIGHTS_SET", LightsBindingSet);
    scene::PointLights::appendShaderDefines(raygenDefines);
    scene::SpotLights::appendShaderDefines(raygenDefines);
    WHEELS_ASSERT(raygenDefines.size() <= raygenDefsLen);

    const size_t anyhitDefsLen = 512;
    String anyhitDefines{scopeAlloc, anyhitDefsLen};
    appendDefineStr(anyhitDefines, "RAY_TRACING_SET", RTBindingSet);
    appendEnumVariantsAsDefines(
        anyhitDefines, "DrawType",
        Span{scene::sDrawTypeNames.data(), scene::sDrawTypeNames.size()});
    appendDefineStr(
        anyhitDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        anyhitDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        anyhitDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(anyhitDefines, "GEOMETRY_SET", GeometryBindingSet);
    appendDefineStr(
        anyhitDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    WHEELS_ASSERT(anyhitDefines.size() <= anyhitDefsLen);

    Optional<gfx::Device::ShaderCompileResult> raygenResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(),
            gfx::Device::CompileShaderModuleArgs{
                .relPath = "shader/rt/direct_illumination/main.rgen",
                .debugName = "restirDiTraceRGEN",
                .defines = raygenDefines,
            });
    Optional<gfx::Device::ShaderCompileResult> rayMissResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(), gfx::Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rmiss",
                                          .debugName = "sceneRMISS",
                                      });
    Optional<gfx::Device::ShaderCompileResult> closestHitResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(), gfx::Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rchit",
                                          .debugName = "sceneRCHIT",
                                      });
    Optional<gfx::Device::ShaderCompileResult> anyHitResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(), gfx::Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rahit",
                                          .debugName = "sceneRAHIT",
                                          .defines = anyhitDefines,
                                      });

    if (raygenResult.has_value() && rayMissResult.has_value() &&
        closestHitResult.has_value() && anyHitResult.has_value())
    {
        destroyShaders();

        m_raygenReflection = WHEELS_MOV(raygenResult->reflection);
        WHEELS_ASSERT(
            sizeof(TracePC) == m_raygenReflection->pushConstantsBytesize());

        m_rayMissReflection = WHEELS_MOV(rayMissResult->reflection);
        WHEELS_ASSERT(
            m_rayMissReflection->pushConstantsBytesize() == 0 ||
            sizeof(TracePC) == m_rayMissReflection->pushConstantsBytesize());

        m_closestHitReflection = WHEELS_MOV(closestHitResult->reflection);
        WHEELS_ASSERT(
            m_closestHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(TracePC) == m_closestHitReflection->pushConstantsBytesize());

        m_anyHitReflection = WHEELS_MOV(anyHitResult->reflection);
        WHEELS_ASSERT(
            m_anyHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(TracePC) == m_anyHitReflection->pushConstantsBytesize());

        m_shaderStages[static_cast<uint32_t>(StageIndex::RayGen)] = {
            .stage = vk::ShaderStageFlagBits::eRaygenKHR,
            .module = raygenResult->module,
            .pName = "main",
        };
        m_shaderStages[static_cast<uint32_t>(StageIndex::Miss)] = {
            .stage = vk::ShaderStageFlagBits::eMissKHR,
            .module = rayMissResult->module,
            .pName = "main",
        };
        m_shaderStages[static_cast<uint32_t>(StageIndex::ClosestHit)] = {
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
            .module = closestHitResult->module,
            .pName = "main",
        };
        m_shaderStages[static_cast<uint32_t>(StageIndex::AnyHit)] = {
            .stage = vk::ShaderStageFlagBits::eAnyHitKHR,
            .module = anyHitResult->module,
            .pName = "main",
        };

        m_shaderGroups[static_cast<uint32_t>(GroupIndex::RayGen)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = static_cast<uint32_t>(StageIndex::RayGen),
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        m_shaderGroups[static_cast<uint32_t>(GroupIndex::Miss)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = static_cast<uint32_t>(StageIndex::Miss),
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        m_shaderGroups[static_cast<uint32_t>(GroupIndex::Hit)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = static_cast<uint32_t>(StageIndex::ClosestHit),
            .anyHitShader = static_cast<uint32_t>(StageIndex::AnyHit),
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        return true;
    }

    if (raygenResult.has_value())
        gfx::gDevice.logical().destroy(raygenResult->module);
    if (rayMissResult.has_value())
        gfx::gDevice.logical().destroy(rayMissResult->module);
    if (closestHitResult.has_value())
        gfx::gDevice.logical().destroy(closestHitResult->module);
    if (anyHitResult.has_value())
        gfx::gDevice.logical().destroy(anyHitResult->module);

    return false;
}

void RtDiTrace::createDescriptorSets(ScopedScratch scopeAlloc)
{
    m_descriptorSetLayout = m_raygenReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), StorageBindingSet,
        vk::ShaderStageFlagBits::eRaygenKHR);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        m_descriptorSetLayout};
    const StaticArray<const char *, MAX_FRAMES_IN_FLIGHT> debugNames{
        "RtDiTrace"};
    gfx::gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_descriptorSets.mut_span());
}

void RtDiTrace::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame, const Input &input,
    ImageHandle illumination)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    WHEELS_ASSERT(m_raygenReflection.has_value());

    const StaticArray descriptorInfos{{
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(input.gbuffer.albedoRoughness)
                    .view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(input.gbuffer.normalMetalness)
                    .view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(input.gbuffer.depth).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(input.reservoirs).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(m_previousIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = gRenderResources.images->resource(illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        gfx::DescriptorInfo{vk::DescriptorImageInfo{
            .sampler = gRenderResources.nearestSampler,
        }},
    }};

    WHEELS_ASSERT(m_raygenReflection.has_value());
    const Array descriptorWrites = m_raygenReflection->generateDescriptorWrites(
        scopeAlloc, StorageBindingSet, m_descriptorSets[nextFrame],
        descriptorInfos);

    gfx::gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void RtDiTrace::createPipeline(
    vk::DescriptorSetLayout camDSLayout,
    const scene::WorldDSLayouts &worldDSLayouts)
{

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[RTBindingSet] = worldDSLayouts.rayTracing;
    setLayouts[StorageBindingSet] = m_descriptorSetLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skybox;
    setLayouts[SceneInstancesBindingSet] = worldDSLayouts.sceneInstances;
    setLayouts[LightsBindingSet] = worldDSLayouts.lights;

    const vk::PushConstantRange pcRange{
        .stageFlags = sVkShaderStageFlagsAllRt,
        .offset = 0,
        .size = sizeof(TracePC),
    };
    m_pipelineLayout = gfx::gDevice.logical().createPipelineLayout(
        vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::RayTracingPipelineCreateInfoKHR pipelineInfo{
        .stageCount = asserted_cast<uint32_t>(m_shaderStages.size()),
        .pStages = m_shaderStages.data(),
        .groupCount = asserted_cast<uint32_t>(m_shaderGroups.size()),
        .pGroups = m_shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .layout = m_pipelineLayout,
    };

    {
        auto pipeline = gfx::gDevice.logical().createRayTracingPipelineKHR(
            vk::DeferredOperationKHR{}, vk::PipelineCache{}, pipelineInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create rt pipeline");

        m_pipeline = pipeline.value;

        gfx::gDevice.logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(m_pipeline)),
                .pObjectName = "RtDiTrace",
            });
    }
}

void RtDiTrace::createShaderBindingTable(ScopedScratch scopeAlloc)
{

    const auto groupCount = asserted_cast<uint32_t>(m_shaderGroups.size());
    const auto groupHandleSize =
        gfx::gDevice.properties().rtPipeline.shaderGroupHandleSize;
    const auto groupBaseAlignment =
        gfx::gDevice.properties().rtPipeline.shaderGroupBaseAlignment;
    m_sbtGroupSize = static_cast<vk::DeviceSize>(roundedUpQuotient(
                         groupHandleSize, groupBaseAlignment)) *
                     groupBaseAlignment;

    const auto sbtSize = groupCount * m_sbtGroupSize;

    Array<uint8_t> shaderHandleStorage{scopeAlloc, sbtSize};
    gfx::checkSuccess(
        gfx::gDevice.logical().getRayTracingShaderGroupHandlesKHR(
            m_pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()),
        "getRayTracingShaderGroupHandlesKHR");

    m_shaderBindingTable = gfx::gDevice.createBuffer(gfx::BufferCreateInfo{
        .desc =
            gfx::BufferDescription{
                .byteSize = sbtSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .cacheDeviceAddress = true,
        .debugName = "RtDiffuseIlluminationSBT",
    });

    auto *pData = static_cast<uint8_t *>(m_shaderBindingTable.mapped);
    for (size_t i = 0; i < groupCount; ++i)
    {
        memcpy(
            pData, shaderHandleStorage.data() + (i * groupHandleSize),
            groupHandleSize);
        pData += m_sbtGroupSize;
    }
}

} // namespace render::rtdi
