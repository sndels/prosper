#include "RTRenderer.hpp"

#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

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
    OutputBindingSet = 2,
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
    uint32_t flags{0};
    uint32_t frameIndex{0};
    float apertureDiameter{0.00001f};
    float focusDistance{1.f};
    float focalLength{0.f};

    struct Flags
    {
        bool skipHistory{false};
        bool accumulate{false};
        bool ibl{false};
        bool depthOfField{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.skipHistory;
    ret |= (uint32_t)flags.accumulate << 1;
    ret |= (uint32_t)flags.ibl << 2;
    ret |= (uint32_t)flags.depthOfField << 3;

    return ret;
}

constexpr std::array<
    const char *, static_cast<size_t>(RTRenderer::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

} // namespace

RTRenderer::RTRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating RTRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RTRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());
}

RTRenderer::~RTRenderer()
{
    if (_device != nullptr)
    {
        destroyPipeline();

        _device->logical().destroy(_descriptorSetLayout);

        _device->destroy(_shaderBindingTable);
        destroyShaders();
    }
}

void RTRenderer::recompileShaders(
    ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
        _accumulationDirty = true;
    }
}

void RTRenderer::drawUi()
{
    auto *currentType = reinterpret_cast<uint32_t *>(&_drawType);
    if (ImGui::BeginCombo("Draw type", sDrawTypeNames[*currentType]))
    {
        for (auto i = 0u;
             i < static_cast<uint32_t>(RTRenderer::DrawType::Count); ++i)
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
    {
        ImGui::Checkbox("Accumulate", &_accumulate);
        _accumulationDirty |= ImGui::Checkbox("Ibl", &_ibl);
    }
}

RTRenderer::Output RTRenderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    bool depthOfField, const vk::Rect2D &renderArea, uint32_t nextFrame,
    bool colorDirty, Profiler *profiler)
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

        if (colorDirty || renderArea.extent.width != previousExtent.width ||
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

        updateDescriptorSet(nextFrame, illumination);

        {
            const StaticArray barriers{{
                _resources->images.transitionBarrier(
                    illumination, ImageState::RayTracingReadWrite),
                _resources->images.transitionBarrier(
                    _previousIllumination, ImageState::RayTracingReadWrite),
            }};

            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount =
                    asserted_cast<uint32_t>(barriers.size()),
                .pImageMemoryBarriers = barriers.data(),
            });
        }

        const auto _s = profiler->createCpuGpuScope(cb, "RT");

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[RTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[OutputBindingSet] = _descriptorSets[nextFrame];
        descriptorSets[MaterialDatasBindingSet] =
            world._materialDatasDSs[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = world._materialTexturesDS;
        descriptorSets[GeometryBindingSet] = world._geometryDS;
        descriptorSets[SkyboxBindingSet] = world._skyboxOnlyDS;
        descriptorSets[ModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSets[nextFrame];
        descriptorSets[LightsBindingSet] =
            scene.lights.descriptorSets[nextFrame];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR, _pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
            .flags = pcFlags(PCBlock::Flags{
                .skipHistory =
                    cam.changedThisFrame() || colorDirty || _accumulationDirty,
                .accumulate = _accumulate,
                .ibl = _ibl,
                .depthOfField = depthOfField,
            }),
            .frameIndex = _frameIndex,
            .apertureDiameter = cam.apertureDiameter(),
            .focusDistance = cam.focusDistance(),
            .focalLength = cam.focalLength(),
        };
        cb.pushConstants(
            _pipelineLayout, sVkShaderStageFlagsAllRt, 0, sizeof(PCBlock),
            &pcBlock);

        const auto sbtAddr =
            _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
                .buffer = _shaderBindingTable.handle,
            });

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

        assert(renderArea.offset.x == 0 && renderArea.offset.y == 0);
        cb.traceRaysKHR(
            &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
            renderArea.extent.width, renderArea.extent.height, 1);

        _resources->images.release(_previousIllumination);
        _previousIllumination = illumination;
        _resources->images.preserve(_previousIllumination);

        // Further passes expect 16bit illumination
        // TODO:
        // Remove this and return the 32bit illumination when shaders are in
        // HLSL and 16bit/32bit texture read layout doesn't matter
        {
            ret.illumination = createIllumination(
                *_resources, renderArea.extent, "illumination");

            {
                const StaticArray barriers{{
                    _resources->images.transitionBarrier(
                        illumination, ImageState::TransferSrc),
                    _resources->images.transitionBarrier(
                        ret.illumination, ImageState::TransferDst),
                }};

                cb.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount =
                        asserted_cast<uint32_t>(barriers.size()),
                    .pImageMemoryBarriers = barriers.data(),
                });
            }

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

void RTRenderer::releasePreserved()
{
    if (_resources->images.isValidHandle(_previousIllumination))
        _resources->images.release(_previousIllumination);
}

void RTRenderer::destroyShaders()
{
    for (auto const &stage : _shaderStages)
        _device->logical().destroyShaderModule(stage.module);
}

void RTRenderer::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

bool RTRenderer::compileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    printf("Compiling RTRenderer shaders\n");

    String raygenDefines{scopeAlloc, 256};
    appendDefineStr(raygenDefines, "NON_UNIFORM_MATERIAL_INDICES");
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
        raygenDefines, "MODEL_INSTANCE_TRFNS_SET",
        ModelInstanceTrfnsBindingSet);
    appendDefineStr(raygenDefines, "LIGHTS_SET", LightsBindingSet);
    PointLights::appendShaderDefines(raygenDefines);
    SpotLights::appendShaderDefines(raygenDefines);

    String anyhitDefines{scopeAlloc, 256};
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

    Optional<Device::ShaderCompileResult> raygenResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rgen",
                                          .debugName = "sceneRGEN",
                                          .defines = raygenDefines,
                                      });
    const Optional<Device::ShaderCompileResult> rayMissResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rmiss",
                                          .debugName = "sceneRMISS",
                                      });
    const Optional<Device::ShaderCompileResult> closestHitResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/rt/scene.rchit",
                                          .debugName = "sceneRCHIT",
                                      });
    const Optional<Device::ShaderCompileResult> anyHitResult =
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
        assert(sizeof(PCBlock) == _raygenReflection->pushConstantsBytesize());

#ifndef NDEBUG
        const ShaderReflection &rayMissReflection = rayMissResult->reflection;
        assert(
            rayMissReflection.pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == rayMissReflection.pushConstantsBytesize());
#endif // !NDEBUG

#ifndef NDEBUG
        const ShaderReflection &closestHitReflection =
            closestHitResult->reflection;
        assert(
            closestHitReflection.pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == closestHitReflection.pushConstantsBytesize());
#endif // !NDEBUG

#ifndef NDEBUG
        const ShaderReflection &anyHitReflection = anyHitResult->reflection;
        assert(
            anyHitReflection.pushConstantsBytesize() == 0 ||
            sizeof(PCBlock) == anyHitReflection.pushConstantsBytesize());
#endif // !NDEBUG

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

void RTRenderer::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _raygenReflection->generateLayoutBindings(
            scopeAlloc, OutputBindingSet, vk::ShaderStageFlagBits::eRaygenKHR);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
}

void RTRenderer::updateDescriptorSet(
    uint32_t nextFrame, ImageHandle illumination)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    assert(_raygenReflection.has_value());

    const StaticArray bindingInfos = {
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView =
                _resources->images.resource(_previousIllumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
        DescriptorInfo{vk::DescriptorImageInfo{
            .imageView = _resources->images.resource(illumination).view,
            .imageLayout = vk::ImageLayout::eGeneral,
        }},
    };

    assert(_raygenReflection.has_value());
    const StaticArray descriptorWrites =
        _raygenReflection->generateDescriptorWrites(
            OutputBindingSet, _descriptorSets[nextFrame], bindingInfos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void RTRenderer::createPipeline(
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[RTBindingSet] = worldDSLayouts.rayTracing;
    setLayouts[OutputBindingSet] = _descriptorSetLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skyboxOnly;
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
                .pObjectName = "RTRenderer",
            });
    }
}

void RTRenderer::createShaderBindingTable(ScopedScratch scopeAlloc)
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
        .debugName = "ShaderBindingTable",
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
