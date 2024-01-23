#include "RtReference.hpp"

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
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

constexpr StaticArray<
    const char *, static_cast<size_t>(RtReference::DrawType::Count)>
    sDrawTypeNames{{DEBUG_DRAW_TYPES_STRS}};

} // namespace

RtReference::RtReference(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const WorldDSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    printf("Creating RtReference\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RtReference shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());
}

RtReference::~RtReference()
{
    if (_device != nullptr)
    {
        destroyPipeline();

        _device->logical().destroy(_descriptorSetLayout);

        _device->destroy(_shaderBindingTable);
        destroyShaders();
    }
}

void RtReference::recompileShaders(
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

void RtReference::drawUi()
{
    _accumulationDirty |= enumDropdown("Draw type", _drawType, sDrawTypeNames);

    if (_drawType == DrawType::Default)
    {
        ImGui::Checkbox("Accumulate", &_accumulate);

        _accumulationDirty |=
            ImGui::Checkbox("Clamp indirect", &_clampIndirect);
        _accumulationDirty |=
            sliderU32("Roulette Start", &_rouletteStartBounce, 0u, _maxBounces);
        _accumulationDirty |=
            sliderU32("Max bounces", &_maxBounces, 1u, sMaxBounces);

        _maxBounces = std::min(_maxBounces, sMaxBounces);
        _rouletteStartBounce = std::min(_rouletteStartBounce, _maxBounces);
    }
}

RtReference::Output RtReference::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
    const Camera &cam, const vk::Rect2D &renderArea, const Options &options,
    uint32_t nextFrame, Profiler *profiler)
{
    _frameIndex = ++_frameIndex % sFramePeriod;

    Output ret;
    {
        // Need 32 bits of precision to accumulate properly
        // TODO:
        // This happens to be the same physical image as last frame for now, but
        // resources should support this kind of accumulation use explicitly
        ImageHandle illumination = _resources->images.create(
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
        if (_resources->images.isValidHandle(_previousIllumination))
            previousExtent =
                _resources->images.resource(_previousIllumination).extent;

        if (options.colorDirty ||
            renderArea.extent.width != previousExtent.width ||
            renderArea.extent.height != previousExtent.height)
        {
            if (_resources->images.isValidHandle(_previousIllumination))
                _resources->images.release(_previousIllumination);

            // Create dummy texture that won't be read from to satisfy binds
            _previousIllumination = _resources->images.create(
                ImageDescription{
                    .format = vk::Format::eR32G32B32A32Sfloat,
                    .width = renderArea.extent.width,
                    .height = renderArea.extent.height,
                    .usageFlags = vk::ImageUsageFlagBits::eStorage |
                                  vk::ImageUsageFlagBits::eTransferSrc |
                                  vk::ImageUsageFlagBits::eSampled,
                },
                "previousRTIllumination");
            _accumulationDirty = true;
        }
        else // We clear debug names each frame
            _resources->images.appendDebugName(
                _previousIllumination, "previousRTIllumination");

        updateDescriptorSet(scopeAlloc.child_scope(), nextFrame, illumination);

        world.currentTLAS().buffer.transition(
            cb, BufferState::RayTracingAccelerationStructureRead);

        transition(
            scopeAlloc.child_scope(), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {illumination, ImageState::RayTracingReadWrite},
                    {_previousIllumination, ImageState::RayTracingReadWrite},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "RtReference");

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, _pipeline);

        const auto &scene = world.currentScene();
        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[RTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[OutputBindingSet] = _descriptorSets[nextFrame];
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
            vk::PipelineBindPoint::eRayTracingKHR, _pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        const CameraParameters &camParams = cam.parameters();

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
            .flags = pcFlags(PCBlock::Flags{
                .skipHistory = cam.changedThisFrame() || options.colorDirty ||
                               _accumulationDirty,
                .accumulate = _accumulate,
                .ibl = options.ibl,
                .depthOfField = options.depthOfField,
                .clampIndirect = _clampIndirect,
            }),
            .frameIndex = _frameIndex,
            .apertureDiameter = camParams.apertureDiameter,
            .focusDistance = camParams.focusDistance,
            .focalLength = camParams.focalLength,
            .rouletteStartBounce = _rouletteStartBounce,
            .maxBounces = _maxBounces,
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

        WHEELS_ASSERT(renderArea.offset.x == 0 && renderArea.offset.y == 0);
        cb.traceRaysKHR(
            &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
            renderArea.extent.width, renderArea.extent.height, 1);

        _resources->images.release(_previousIllumination);
        _previousIllumination = illumination;
        _resources->images.preserve(_previousIllumination);

        // Further passes expect 16bit illumination with pipelines created with
        // the attachment format
        {
            ret.illumination = createIllumination(
                *_resources, renderArea.extent, "illumination");

            transition(
                WHEELS_MOV(scopeAlloc), *_resources, cb,
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

void RtReference::releasePreserved()
{
    if (_resources->images.isValidHandle(_previousIllumination))
        _resources->images.release(_previousIllumination);
}

void RtReference::destroyShaders()
{
    for (auto const &stage : _shaderStages)
        _device->logical().destroyShaderModule(stage.module);
}

void RtReference::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

bool RtReference::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    printf("Compiling RtReference shaders\n");

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
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/rt/reference/main.rgen",
                .debugName = "referenceRGEN",
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

void RtReference::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    _descriptorSetLayout = _raygenReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), *_device, OutputBindingSet,
        vk::ShaderStageFlagBits::eRaygenKHR);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void RtReference::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame, ImageHandle illumination)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    WHEELS_ASSERT(_raygenReflection.has_value());

    const StaticArray descriptorInfos{{
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(_previousIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
    }};

    WHEELS_ASSERT(_raygenReflection.has_value());
    const Array descriptorWrites = _raygenReflection->generateDescriptorWrites(
        scopeAlloc, OutputBindingSet, _descriptorSets[nextFrame],
        descriptorInfos);

    _device->logical().updateDescriptorSets(
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
    setLayouts[OutputBindingSet] = _descriptorSetLayout;
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
                .pObjectName = "RtReference",
            });
    }
}

void RtReference::createShaderBindingTable(ScopedScratch scopeAlloc)
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
        .cacheDeviceAddress = true,
        .debugName = "RtReferenceSBT",
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
