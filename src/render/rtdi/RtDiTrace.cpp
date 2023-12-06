#include "RtDiTrace.hpp"

#include "../../gfx/DescriptorAllocator.hpp"
#include "../../gfx/VkUtils.hpp"
#include "../../scene/Camera.hpp"
#include "../../scene/Scene.hpp"
#include "../../scene/World.hpp"
#include "../../utils/Profiler.hpp"
#include "../../utils/Utils.hpp"
#include "../GBufferRenderer.hpp"
#include "../RenderResources.hpp"
#include "../RenderTargets.hpp"

#include <imgui.h>

using namespace wheels;

// Based on RT Gems II chapter 16

namespace
{

constexpr uint32_t sFramePeriod = 4096;

enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    RTBindingSet = 1,
    StorageBindingSet = 2,
    MaterialDatasBindingSet = 3,
    MaterialTexturesBindingSet = 4,
    GeometryBindingSet = 5,
    SkyboxBindingSet = 6,
    ModelInstanceTrfnsBindingSet = 7,
    LightsBindingSet = 8,
    BindingSetCount = 9,
};

constexpr vk::ShaderStageFlags sVkShaderStageFlagsAllRt =
    vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eAnyHitKHR |
    vk::ShaderStageFlagBits::eClosestHitKHR |
    vk::ShaderStageFlagBits::eMissKHR |
    vk::ShaderStageFlagBits::eIntersectionKHR;

enum class StageIndex : uint32_t
{
    RayGen = 0,
    ClosestHit,
    AnyHit,
    Miss,
};

enum class GroupIndex : uint32_t
{
    RayGen = 0,
    Hit,
    Miss,
};

struct PCBlock
{
    uint32_t drawType{0};
    uint32_t frameIndex{0};
    uint32_t flags{0};

    struct Flags
    {
        bool skipHistory{false};
        bool accumulate{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.skipHistory;
    ret |= (uint32_t)flags.accumulate << 1;

    return ret;
}

constexpr std::array<
    const char *, static_cast<size_t>(RtDiTrace::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, const GBufferRendererOutput &gbuffer)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(gbuffer.albedoRoughness).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

} // namespace

RtDiTrace::RtDiTrace(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    printf("Creating RtDiTrace\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RtDiTrace shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());
}

RtDiTrace::~RtDiTrace()
{
    if (_device != nullptr)
    {
        destroyPipeline();

        _device->logical().destroy(_descriptorSetLayout);

        _device->destroy(_shaderBindingTable);
        destroyShaders();
    }
}

void RtDiTrace::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_raygenReflection.has_value());
    WHEELS_ASSERT(_rayMissReflection.has_value());
    WHEELS_ASSERT(_closestHitReflection.has_value());
    WHEELS_ASSERT(_anyHitReflection.has_value());
    if (!_raygenReflection->affected(changedFiles) &&
        !_rayMissReflection->affected(changedFiles) &&
        !_closestHitReflection->affected(changedFiles) &&
        !_anyHitReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
        _accumulationDirty = true;
    }
}

void RtDiTrace::drawUi()
{
    auto *currentType = reinterpret_cast<uint32_t *>(&_drawType);
    if (ImGui::BeginCombo("Draw type", sDrawTypeNames[*currentType]))
    {
        for (auto i = 0u; i < static_cast<uint32_t>(RtDiTrace::DrawType::Count);
             ++i)
        {
            bool selected = *currentType == i;
            if (ImGui::Selectable(sDrawTypeNames[i], &selected))
            {
                _drawType = static_cast<DrawType>(i);
                _accumulationDirty = true;
            }
        }
        ImGui::EndCombo();
    }

    if (_drawType == DrawType::Default)
        ImGui::Checkbox("Accumulate", &_accumulate);
}

RtDiTrace::Output RtDiTrace::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const Input &input, bool resetAccumulation,
    uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    _frameIndex = ++_frameIndex % sFramePeriod;

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.gbuffer);

        const ImageDescription accumulateImageDescription = ImageDescription{
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
        ImageHandle illumination = _resources->images.create(
            accumulateImageDescription, "RtDiTrace32bit");

        vk::Extent3D previousExtent;
        if (_resources->images.isValidHandle(_previousIllumination))
            previousExtent =
                _resources->images.resource(_previousIllumination).extent;

        if (resetAccumulation || renderExtent.width != previousExtent.width ||
            renderExtent.height != previousExtent.height)
        {
            if (_resources->images.isValidHandle(_previousIllumination))
                _resources->images.release(_previousIllumination);

            // Create dummy texture that won't be read from to satisfy binds
            _previousIllumination = _resources->images.create(
                accumulateImageDescription, "previousRtDiTrace");
            _accumulationDirty = true;
        }
        else // We clear debug names each frame
            _resources->images.appendDebugName(
                _previousIllumination, "previousRtDiTrace");

        updateDescriptorSet(
            WHEELS_MOV(scopeAlloc), nextFrame, input, illumination);

        {
            const vk::MemoryBarrier2 barrier{
                .srcStageMask =
                    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                .srcAccessMask =
                    vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                .dstStageMask =
                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                .dstAccessMask =
                    vk::AccessFlagBits2::eAccelerationStructureReadKHR,
            };
            cb.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
            });
        }

        transition<6>(
            *_resources, cb,
            {
                {input.gbuffer.albedoRoughness, ImageState::RayTracingRead},
                {input.gbuffer.normalMetalness, ImageState::RayTracingRead},
                {input.gbuffer.depth, ImageState::RayTracingRead},
                {input.reservoirs, ImageState::RayTracingRead},
                {_previousIllumination, ImageState::RayTracingRead},
                {illumination, ImageState::RayTracingReadWrite},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "  Trace");

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, _pipeline);

        const auto &scene = world.currentScene();
        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[RTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[StorageBindingSet] = _descriptorSets[nextFrame];
        descriptorSets[MaterialDatasBindingSet] =
            worldDSes.materialDatas[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
        descriptorSets[GeometryBindingSet] = worldDSes.geometry;
        descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
        descriptorSets[ModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSet;
        descriptorSets[LightsBindingSet] = worldDSes.lights;

        const StaticArray dynamicOffsets{
            cam.bufferOffset(),
            worldByteOffsets.modelInstanceTransforms,
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
        };

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR, _pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
            .frameIndex = _frameIndex,
            .flags = pcFlags(PCBlock::Flags{
                .skipHistory = cam.changedThisFrame() || resetAccumulation ||
                               _accumulationDirty,
                .accumulate = _accumulate,
            }),
        };
        cb.pushConstants(
            _pipelineLayout, sVkShaderStageFlagsAllRt, 0, sizeof(PCBlock),
            &pcBlock);

        WHEELS_ASSERT(_shaderBindingTable.deviceAddress != 0);
        const vk::DeviceAddress sbtAddr = _shaderBindingTable.deviceAddress;

        const vk::StridedDeviceAddressRegionKHR rayGenRegion{
            .deviceAddress = sbtAddr + _sbtGroupSize * static_cast<uint32_t>(
                                                           GroupIndex::RayGen),
            .stride = _sbtGroupSize,
            .size = _sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR missRegion{
            .deviceAddress = sbtAddr + _sbtGroupSize * static_cast<uint32_t>(
                                                           GroupIndex::Miss),
            .stride = _sbtGroupSize,
            .size = _sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR hitRegion{
            .deviceAddress = sbtAddr + _sbtGroupSize * static_cast<uint32_t>(
                                                           GroupIndex::Hit),
            .stride = _sbtGroupSize,
            .size = _sbtGroupSize,
        };

        const vk::StridedDeviceAddressRegionKHR callableRegion;

        cb.traceRaysKHR(
            &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
            renderExtent.width, renderExtent.height, 1);

        _resources->images.release(_previousIllumination);
        _previousIllumination = illumination;
        _resources->images.preserve(_previousIllumination);

        // Further passes expect 16bit illumination
        // TODO:
        // Remove this and return the 32bit illumination when shaders are in
        // HLSL and 16bit/32bit texture read layout doesn't matter
        {
            ret.illumination =
                createIllumination(*_resources, renderExtent, "RtDiTrace");

            transition<2>(
                *_resources, cb,
                {
                    {illumination, ImageState::TransferSrc},
                    {ret.illumination, ImageState::TransferDst},
                });

            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};

            const std::array offsets{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{
                    asserted_cast<int32_t>(renderExtent.width),
                    asserted_cast<int32_t>(renderExtent.height),
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
                _resources->images.nativeHandle(illumination),
                vk::ImageLayout::eTransferSrcOptimal,
                _resources->images.nativeHandle(ret.illumination),
                vk::ImageLayout::eTransferDstOptimal, 1, &blit,
                vk::Filter::eLinear);
        }
    }

    _accumulationDirty = false;

    return ret;
}

void RtDiTrace::releasePreserved()
{
    if (_resources->images.isValidHandle(_previousIllumination))
        _resources->images.release(_previousIllumination);
}

void RtDiTrace::destroyShaders()
{
    for (auto const &stage : _shaderStages)
        _device->logical().destroyShaderModule(stage.module);
}

void RtDiTrace::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

bool RtDiTrace::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    printf("Compiling RtDiTrace shaders\n");

    const size_t raygenDefsLen = 768;
    String raygenDefines{scopeAlloc, raygenDefsLen};
    appendDefineStr(raygenDefines, "NON_UNIFORM_MATERIAL_INDICES");
    appendDefineStr(raygenDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(raygenDefines, "RAY_TRACING_SET", RTBindingSet);
    appendDefineStr(raygenDefines, "STORAGE_SET", StorageBindingSet);
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
        raygenDefines, "MODEL_INSTANCE_TRFNS_SET",
        ModelInstanceTrfnsBindingSet);
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
        anyhitDefines, "MODEL_INSTANCE_TRFNS_SET",
        ModelInstanceTrfnsBindingSet);
    WHEELS_ASSERT(anyhitDefines.size() <= anyhitDefsLen);

    Optional<Device::ShaderCompileResult> raygenResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/rt/direct_illumination/main.rgen",
                .debugName = "restirDiTraceRGEN",
                .defines = raygenDefines,
            });
    Optional<Device::ShaderCompileResult> rayMissResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rmiss",
                                          .debugName = "sceneRMISS",
                                      });
    Optional<Device::ShaderCompileResult> closestHitResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rchit",
                                          .debugName = "sceneRCHIT",
                                      });
    Optional<Device::ShaderCompileResult> anyHitResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rahit",
                                          .debugName = "sceneRAHIT",
                                          .defines = anyhitDefines,
                                      });

    if (raygenResult.has_value() && rayMissResult.has_value() &&
        closestHitResult.has_value() && anyHitResult.has_value())
    {
        destroyShaders();

        _raygenReflection = WHEELS_MOV(raygenResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _raygenReflection->pushConstantsBytesize());

        _rayMissReflection = WHEELS_MOV(rayMissResult->reflection);
        WHEELS_ASSERT(
            _rayMissReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == _rayMissReflection->pushConstantsBytesize());

        _closestHitReflection = WHEELS_MOV(closestHitResult->reflection);
        WHEELS_ASSERT(
            _closestHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == _closestHitReflection->pushConstantsBytesize());

        _anyHitReflection = WHEELS_MOV(anyHitResult->reflection);
        WHEELS_ASSERT(
            _anyHitReflection->pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == _anyHitReflection->pushConstantsBytesize());

        _shaderStages[static_cast<uint32_t>(StageIndex::RayGen)] = {
            .stage = vk::ShaderStageFlagBits::eRaygenKHR,
            .module = raygenResult->module,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::Miss)] = {
            .stage = vk::ShaderStageFlagBits::eMissKHR,
            .module = rayMissResult->module,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::ClosestHit)] = {
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
            .module = closestHitResult->module,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::AnyHit)] = {
            .stage = vk::ShaderStageFlagBits::eAnyHitKHR,
            .module = anyHitResult->module,
            .pName = "main",
        };

        _shaderGroups[static_cast<uint32_t>(GroupIndex::RayGen)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = static_cast<uint32_t>(StageIndex::RayGen),
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        _shaderGroups[static_cast<uint32_t>(GroupIndex::Miss)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = static_cast<uint32_t>(StageIndex::Miss),
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        _shaderGroups[static_cast<uint32_t>(GroupIndex::Hit)] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = static_cast<uint32_t>(StageIndex::ClosestHit),
            .anyHitShader = static_cast<uint32_t>(StageIndex::AnyHit),
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        return true;
    }

    if (raygenResult.has_value())
        _device->logical().destroy(raygenResult->module);
    if (rayMissResult.has_value())
        _device->logical().destroy(rayMissResult->module);
    if (closestHitResult.has_value())
        _device->logical().destroy(closestHitResult->module);
    if (anyHitResult.has_value())
        _device->logical().destroy(anyHitResult->module);

    return false;
}

void RtDiTrace::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    _descriptorSetLayout = _raygenReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), *_device, StorageBindingSet,
        vk::ShaderStageFlagBits::eRaygenKHR);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void RtDiTrace::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame, const Input &input,
    ImageHandle illumination)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    WHEELS_ASSERT(_raygenReflection.has_value());

    const StaticArray descriptorInfos{
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.gbuffer.albedoRoughness).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(input.gbuffer.normalMetalness).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(input.gbuffer.depth).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(input.reservoirs).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(_previousIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .sampler = _resources->nearestSampler,
        }},
    };

    WHEELS_ASSERT(_raygenReflection.has_value());
    const Array descriptorWrites = _raygenReflection->generateDescriptorWrites(
        scopeAlloc, StorageBindingSet, _descriptorSets[nextFrame],
        descriptorInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void RtDiTrace::createPipeline(
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
{

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[RTBindingSet] = worldDSLayouts.rayTracing;
    setLayouts[StorageBindingSet] = _descriptorSetLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skybox;
    setLayouts[ModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;
    setLayouts[LightsBindingSet] = worldDSLayouts.lights;

    const vk::PushConstantRange pcRange{
        .stageFlags = sVkShaderStageFlagsAllRt,
        .offset = 0,
        .size = sizeof(PCBlock),
    };
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::RayTracingPipelineCreateInfoKHR pipelineInfo{
        .stageCount = asserted_cast<uint32_t>(_shaderStages.size()),
        .pStages = _shaderStages.data(),
        .groupCount = asserted_cast<uint32_t>(_shaderGroups.size()),
        .pGroups = _shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .layout = _pipelineLayout,
    };

    {
        auto pipeline = _device->logical().createRayTracingPipelineKHR(
            vk::DeferredOperationKHR{}, vk::PipelineCache{}, pipelineInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create rt pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "RtDiTrace",
            });
    }
}

void RtDiTrace::createShaderBindingTable(ScopedScratch scopeAlloc)
{

    const auto groupCount = asserted_cast<uint32_t>(_shaderGroups.size());
    const auto groupHandleSize =
        _device->properties().rtPipeline.shaderGroupHandleSize;
    const auto groupBaseAlignment =
        _device->properties().rtPipeline.shaderGroupBaseAlignment;
    _sbtGroupSize = static_cast<vk::DeviceSize>(
                        ((groupHandleSize - 1) / groupBaseAlignment) + 1) *
                    groupBaseAlignment;

    const auto sbtSize = groupCount * _sbtGroupSize;

    Array<uint8_t> shaderHandleStorage{scopeAlloc, sbtSize};
    checkSuccess(
        _device->logical().getRayTracingShaderGroupHandlesKHR(
            _pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()),
        "getRayTracingShaderGroupHandlesKHR");

    _shaderBindingTable = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sbtSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .cacheDeviceAddress = true,
        .debugName = "RtDiffuseIlluminationSBT",
    });

    auto *pData = reinterpret_cast<uint8_t *>(_shaderBindingTable.mapped);
    for (size_t i = 0; i < groupCount; ++i)
    {
        memcpy(
            pData, shaderHandleStorage.data() + i * groupHandleSize,
            groupHandleSize);
        pData += _sbtGroupSize;
    }
}
