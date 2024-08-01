#include "RtReference.hpp"

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Logger.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Ui.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

#include <imgui.h>

using namespace wheels;

// Based on RT Gems II chapter 16

namespace
{

constexpr uint32_t sFramePeriod = 4096;

enum BindingSet : uint32_t
{
    CameraBindingSet,
    RTBindingSet,
    OutputBindingSet,
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

enum class StageIndex : uint32_t
{
    RayGen,
    ClosestHit,
    AnyHit,
    Miss,
};

enum class GroupIndex : uint32_t
{
    RayGen,
    Hit,
    Miss,
};

struct PCBlock
{
    uint32_t drawType{0};
    uint32_t flags{0};
    uint32_t frameIndex{0};
    float apertureDiameter{0.00001f};
    float focusDistance{1.f};
    float focalLength{0.f};
    uint32_t rouletteStartBounce{3};
    uint32_t maxBounces{RtReference::sMaxBounces};

    struct Flags
    {
        bool skipHistory{false};
        bool accumulate{false};
        bool ibl{false};
        bool depthOfField{false};
        bool clampIndirect{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.skipHistory;
    ret |= (uint32_t)flags.accumulate << 1;
    ret |= (uint32_t)flags.ibl << 2;
    ret |= (uint32_t)flags.depthOfField << 3;
    ret |= (uint32_t)flags.clampIndirect << 4;

    return ret;
}

} // namespace

RtReference::~RtReference()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    destroyPipeline();

    gDevice.logical().destroy(m_descriptorSetLayout);

    gDevice.destroy(m_shaderBindingTable);
    destroyShaders();
}

void RtReference::init(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating RtReference");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RtReference shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope());
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());

    m_initialized = true;
}

void RtReference::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
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

void RtReference::drawUi()
{
    WHEELS_ASSERT(m_initialized);

    ImGui::Checkbox("Accumulate", &m_accumulate);

    m_accumulationDirty |= ImGui::Checkbox("Clamp indirect", &m_clampIndirect);
    m_accumulationDirty |=
        sliderU32("Roulette Start", &m_rouletteStartBounce, 0u, m_maxBounces);
    m_accumulationDirty |=
        sliderU32("Max bounces", &m_maxBounces, 1u, sMaxBounces);

    m_maxBounces = std::min(m_maxBounces, sMaxBounces);
    m_rouletteStartBounce = std::min(m_rouletteStartBounce, m_maxBounces);
}

RtReference::Output RtReference::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
    const Camera &cam, const vk::Rect2D &renderArea, const Options &options,
    uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("RtReference");

    m_frameIndex = (m_frameIndex + 1) % sFramePeriod;

    Output ret;
    {
        // Need 32 bits of precision to accumulate properly
        // TODO:
        // This happens to be the same physical image as last frame for now, but
        // resources should support this kind of accumulation use explicitly
        ImageHandle illumination = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR32G32B32A32Sfloat,
                .width = renderArea.extent.width,
                .height = renderArea.extent.height,
                .usageFlags = vk::ImageUsageFlagBits::eStorage |
                              vk::ImageUsageFlagBits::eTransferSrc |
                              vk::ImageUsageFlagBits::eSampled,
            },
            "rtIllumination");

        vk::Extent3D previousExtent;
        if (gRenderResources.images->isValidHandle(m_previousIllumination))
            previousExtent =
                gRenderResources.images->resource(m_previousIllumination)
                    .extent;

        if (options.colorDirty ||
            renderArea.extent.width != previousExtent.width ||
            renderArea.extent.height != previousExtent.height)
        {
            if (gRenderResources.images->isValidHandle(m_previousIllumination))
                gRenderResources.images->release(m_previousIllumination);

            // Create dummy texture that won't be read from to satisfy binds
            m_previousIllumination = gRenderResources.images->create(
                ImageDescription{
                    .format = vk::Format::eR32G32B32A32Sfloat,
                    .width = renderArea.extent.width,
                    .height = renderArea.extent.height,
                    .usageFlags = vk::ImageUsageFlagBits::eStorage |
                                  vk::ImageUsageFlagBits::eTransferSrc |
                                  vk::ImageUsageFlagBits::eSampled,
                },
                "previousRTIllumination");
            m_accumulationDirty = true;
        }
        else // We clear debug names each frame
            gRenderResources.images->appendDebugName(
                m_previousIllumination, "previousRTIllumination");

        updateDescriptorSet(scopeAlloc.child_scope(), nextFrame, illumination);

        world.currentTLAS().buffer.transition(
            cb, BufferState::RayTracingAccelerationStructureRead);

        transition(
            scopeAlloc.child_scope(), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {illumination, ImageState::RayTracingReadWrite},
                    {m_previousIllumination, ImageState::RayTracingReadWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "RtReference");

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_pipeline);

        const auto &scene = world.currentScene();
        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[RTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[OutputBindingSet] = m_descriptorSets[nextFrame];
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

        const CameraParameters &camParams = cam.parameters();

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(options.drawType),
            .flags = pcFlags(PCBlock::Flags{
                .skipHistory = cam.changedThisFrame() || options.colorDirty ||
                               m_accumulationDirty,
                .accumulate = m_accumulate,
                .ibl = options.ibl,
                .depthOfField = options.depthOfField,
                .clampIndirect = m_clampIndirect,
            }),
            .frameIndex = m_frameIndex,
            .apertureDiameter = camParams.apertureDiameter,
            .focusDistance = camParams.focusDistance,
            .focalLength = camParams.focalLength,
            .rouletteStartBounce = m_rouletteStartBounce,
            .maxBounces = m_maxBounces,
        };
        cb.pushConstants(
            m_pipelineLayout, sVkShaderStageFlagsAllRt, 0, sizeof(PCBlock),
            &pcBlock);

        WHEELS_ASSERT(m_shaderBindingTable.deviceAddress != 0);
        const vk::DeviceAddress sbtAddr = m_shaderBindingTable.deviceAddress;

        const vk::StridedDeviceAddressRegionKHR rayGenRegion{
            .deviceAddress = sbtAddr + m_sbtGroupSize * static_cast<uint32_t>(
                                                            GroupIndex::RayGen),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR missRegion{
            .deviceAddress = sbtAddr + m_sbtGroupSize * static_cast<uint32_t>(
                                                            GroupIndex::Miss),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR hitRegion{
            .deviceAddress = sbtAddr + m_sbtGroupSize * static_cast<uint32_t>(
                                                            GroupIndex::Hit),
            .stride = m_sbtGroupSize,
            .size = m_sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR callableRegion;

        WHEELS_ASSERT(renderArea.offset.x == 0 && renderArea.offset.y == 0);
        cb.traceRaysKHR(
            &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
            renderArea.extent.width, renderArea.extent.height, 1);

        gRenderResources.images->release(m_previousIllumination);
        m_previousIllumination = illumination;
        gRenderResources.images->preserve(m_previousIllumination);

        // Further passes expect 16bit illumination with pipelines created with
        // the attachment format
        {
            ret.illumination =
                createIllumination(renderArea.extent, "illumination");

            transition(
                WHEELS_MOV(scopeAlloc), cb,
                Transitions{
                    .images = StaticArray<ImageTransition, 2>{{
                        {illumination, ImageState::TransferSrc},
                        {ret.illumination, ImageState::TransferDst},
                    }},
                });

            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};

            const std::array offsets{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{
                    asserted_cast<int32_t>(renderArea.extent.width),
                    asserted_cast<int32_t>(renderArea.extent.height),
                    1,
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

void RtReference::releasePreserved()
{
    WHEELS_ASSERT(m_initialized);

    if (gRenderResources.images->isValidHandle(m_previousIllumination))
        gRenderResources.images->release(m_previousIllumination);
}

void RtReference::destroyShaders()
{
    for (auto const &stage : m_shaderStages)
        gDevice.logical().destroyShaderModule(stage.module);
}

void RtReference::destroyPipeline()
{
    gDevice.logical().destroy(m_pipeline);
    gDevice.logical().destroy(m_pipelineLayout);
}

bool RtReference::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    const size_t raygenDefsLen = 768;
    String raygenDefines{scopeAlloc, raygenDefsLen};
    appendDefineStr(raygenDefines, "MAX_BOUNCES", sMaxBounces);
    appendDefineStr(raygenDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(raygenDefines, "RAY_TRACING_SET", RTBindingSet);
    appendDefineStr(raygenDefines, "OUTPUT_SET", OutputBindingSet);
    appendEnumVariantsAsDefines(
        raygenDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
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
    PointLights::appendShaderDefines(raygenDefines);
    SpotLights::appendShaderDefines(raygenDefines);
    WHEELS_ASSERT(raygenDefines.size() <= raygenDefsLen);

    const size_t anyhitDefsLen = 512;
    String anyhitDefines{scopeAlloc, anyhitDefsLen};
    appendDefineStr(anyhitDefines, "RAY_TRACING_SET", RTBindingSet);
    appendEnumVariantsAsDefines(
        anyhitDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
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

    Optional<Device::ShaderCompileResult> raygenResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/rt/reference/main.rgen",
                .debugName = "referenceRGEN",
                .defines = raygenDefines,
            });
    Optional<Device::ShaderCompileResult> rayMissResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rmiss",
                                          .debugName = "sceneRMISS",
                                      });
    Optional<Device::ShaderCompileResult> closestHitResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rchit",
                                          .debugName = "sceneRCHIT",
                                      });
    Optional<Device::ShaderCompileResult> anyHitResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
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
            sizeof(PCBlock) == m_raygenReflection->pushConstantsBytesize());

        m_rayMissReflection = WHEELS_MOV(rayMissResult->reflection);
        WHEELS_ASSERT(
            m_rayMissReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == m_rayMissReflection->pushConstantsBytesize());

        m_closestHitReflection = WHEELS_MOV(closestHitResult->reflection);
        WHEELS_ASSERT(
            m_closestHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == m_closestHitReflection->pushConstantsBytesize());

        m_anyHitReflection = WHEELS_MOV(anyHitResult->reflection);
        WHEELS_ASSERT(
            m_anyHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == m_anyHitReflection->pushConstantsBytesize());

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
        gDevice.logical().destroy(raygenResult->module);
    if (rayMissResult.has_value())
        gDevice.logical().destroy(rayMissResult->module);
    if (closestHitResult.has_value())
        gDevice.logical().destroy(closestHitResult->module);
    if (anyHitResult.has_value())
        gDevice.logical().destroy(anyHitResult->module);

    return false;
}

void RtReference::createDescriptorSets(ScopedScratch scopeAlloc)
{
    m_descriptorSetLayout = m_raygenReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), OutputBindingSet,
        vk::ShaderStageFlagBits::eRaygenKHR);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        m_descriptorSetLayout};
    const StaticArray<const char *, MAX_FRAMES_IN_FLIGHT> debugNames{
        "RtReference"};
    gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_descriptorSets.mut_span());
}

void RtReference::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame, ImageHandle illumination)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    WHEELS_ASSERT(m_raygenReflection.has_value());

    const StaticArray descriptorInfos{{
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                gRenderResources.images->resource(m_previousIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = gRenderResources.images->resource(illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
    }};

    WHEELS_ASSERT(m_raygenReflection.has_value());
    const Array descriptorWrites = m_raygenReflection->generateDescriptorWrites(
        scopeAlloc, OutputBindingSet, m_descriptorSets[nextFrame],
        descriptorInfos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void RtReference::createPipeline(
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[RTBindingSet] = worldDSLayouts.rayTracing;
    setLayouts[OutputBindingSet] = m_descriptorSetLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skybox;
    setLayouts[SceneInstancesBindingSet] = worldDSLayouts.sceneInstances;
    setLayouts[LightsBindingSet] = worldDSLayouts.lights;

    const vk::PushConstantRange pcRange{
        .stageFlags = sVkShaderStageFlagsAllRt,
        .offset = 0,
        .size = sizeof(PCBlock),
    };
    m_pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
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
        auto pipeline = gDevice.logical().createRayTracingPipelineKHR(
            vk::DeferredOperationKHR{}, vk::PipelineCache{}, pipelineInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create rt pipeline");

        m_pipeline = pipeline.value;

        gDevice.logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(m_pipeline)),
                .pObjectName = "RtReference",
            });
    }
}

void RtReference::createShaderBindingTable(ScopedScratch scopeAlloc)
{

    const auto groupCount = asserted_cast<uint32_t>(m_shaderGroups.size());
    const auto groupHandleSize =
        gDevice.properties().rtPipeline.shaderGroupHandleSize;
    const auto groupBaseAlignment =
        gDevice.properties().rtPipeline.shaderGroupBaseAlignment;
    m_sbtGroupSize = static_cast<vk::DeviceSize>(roundedUpQuotient(
                         groupHandleSize, groupBaseAlignment)) *
                     groupBaseAlignment;

    const auto sbtSize = groupCount * m_sbtGroupSize;

    Array<uint8_t> shaderHandleStorage{scopeAlloc, sbtSize};
    checkSuccess(
        gDevice.logical().getRayTracingShaderGroupHandlesKHR(
            m_pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()),
        "getRayTracingShaderGroupHandlesKHR");

    m_shaderBindingTable = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sbtSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .cacheDeviceAddress = true,
        .debugName = "RtReferenceSBT",
    });

    auto *pData = static_cast<uint8_t *>(m_shaderBindingTable.mapped);
    for (size_t i = 0; i < groupCount; ++i)
    {
        memcpy(
            pData, shaderHandleStorage.data() + i * groupHandleSize,
            groupHandleSize);
        pData += m_sbtGroupSize;
    }
}
