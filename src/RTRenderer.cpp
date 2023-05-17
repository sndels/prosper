#include "RTRenderer.hpp"

#include "Utils.hpp"
#include "VkUtils.hpp"

#include <imgui.h>

using namespace wheels;

// Based on RT Gems II chapter 16

namespace
{

constexpr uint32_t sFramePeriod = 4096;

constexpr uint32_t sCameraBindingSet = 0;
constexpr uint32_t sRTBindingSet = 1;
constexpr uint32_t sOutputBindingSet = 2;
constexpr uint32_t sMaterialsBindingSet = 3;
constexpr uint32_t sGeometryBindingSet = 4;
constexpr uint32_t sSkyboxBindingSet = 5;
constexpr uint32_t sModelInstanceTrfnsBindingSet = 6;
constexpr uint32_t sLightsBindingSet = 7;

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

    struct Flags
    {
        bool colorDirty{false};
        bool accumulate{false};
        bool ibl{false};
    };
};

uint32_t pcFlags(PCBlock::Flags flags)
{
    uint32_t ret = 0;

    ret |= (uint32_t)flags.colorDirty;
    ret |= (uint32_t)flags.accumulate << 1;
    ret |= (uint32_t)flags.ibl << 2;

    return ret;
}

constexpr std::array<
    const char *, static_cast<size_t>(RTRenderer::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

} // namespace

RTRenderer::RTRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating RTRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("RTRenderer shader compilation failed");

    createDescriptorSets();
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable(scopeAlloc.child_scope());

    recreate();
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

void RTRenderer::recreate()
{
    updateDescriptorSets();
    _accumulationDirty = true;
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

void RTRenderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, uint32_t nextFrame, bool colorDirty,
    Profiler *profiler)
{
    _frameIndex = ++_frameIndex % sFramePeriod;

    {
        const auto _s = profiler->createCpuGpuScope(cb, "RT");

        _resources->images.sceneColor.transition(
            cb,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                .accessMask = vk::AccessFlagBits2::eShaderStorageWrite |
                              vk::AccessFlagBits2::eShaderStorageRead,
                .layout = vk::ImageLayout::eGeneral,
            });

        cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, 8> descriptorSets{VK_NULL_HANDLE};
        descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[sRTBindingSet] = scene.rtDescriptorSet;
        descriptorSets[sOutputBindingSet] = _descriptorSets[nextFrame];
        descriptorSets[sMaterialsBindingSet] = world._materialTexturesDS;
        descriptorSets[sGeometryBindingSet] = world._geometryDS;
        descriptorSets[sSkyboxBindingSet] = world._skyboxOnlyDS;
        descriptorSets[sModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSets[nextFrame];
        descriptorSets[sLightsBindingSet] =
            scene.lights.descriptorSets[nextFrame];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR, _pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
            .flags = pcFlags(PCBlock::Flags{
                .colorDirty =
                    cam.changedThisFrame() || colorDirty || _accumulationDirty,
                .accumulate = _accumulate,
                .ibl = _ibl,
            }),
            .frameIndex = _frameIndex,
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
    }

    _accumulationDirty = false;
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
    appendDefineStr(raygenDefines, "CAMERA_SET", sCameraBindingSet);
    appendDefineStr(raygenDefines, "RAY_TRACING_SET", sRTBindingSet);
    appendDefineStr(raygenDefines, "OUTPUT_SET", sOutputBindingSet);
    appendEnumVariantsAsDefines(
        raygenDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    appendDefineStr(raygenDefines, "MATERIALS_SET", sMaterialsBindingSet);
    appendDefineStr(
        raygenDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(raygenDefines, "GEOMETRY_SET", sGeometryBindingSet);
    appendDefineStr(raygenDefines, "SKYBOX_SET", sSkyboxBindingSet);
    appendDefineStr(
        raygenDefines, "MODEL_INSTANCE_TRFNS_SET",
        sModelInstanceTrfnsBindingSet);
    appendDefineStr(raygenDefines, "LIGHTS_SET", sLightsBindingSet);
    PointLights::appendShaderDefines(raygenDefines);
    SpotLights::appendShaderDefines(raygenDefines);

    String anyhitDefines{scopeAlloc, 256};
    appendDefineStr(anyhitDefines, "RAY_TRACING_SET", sRTBindingSet);
    appendEnumVariantsAsDefines(
        anyhitDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    appendDefineStr(anyhitDefines, "MATERIALS_SET", sMaterialsBindingSet);
    appendDefineStr(
        anyhitDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(anyhitDefines, "GEOMETRY_SET", sGeometryBindingSet);
    appendDefineStr(
        anyhitDefines, "MODEL_INSTANCE_TRFNS_SET",
        sModelInstanceTrfnsBindingSet);

    const auto raygenSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/rt/scene.rgen",
                                      .debugName = "sceneRGEN",
                                      .defines = raygenDefines,
                                  });
    const auto rayMissSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/rt/scene.rmiss",
                                      .debugName = "sceneRMISS",
                                  });
    const auto closestHitSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/rt/scene.rchit",
                                      .debugName = "sceneRCHIT",
                                  });
    const auto anyHitSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/rt/scene.rahit",
                                      .debugName = "sceneRAHIT",
                                      .defines = anyhitDefines,
                                  });

    if (raygenSM.has_value() && rayMissSM.has_value() &&
        closestHitSM.has_value() && anyHitSM.has_value())
    {
        destroyShaders();

        _shaderStages[static_cast<uint32_t>(StageIndex::RayGen)] = {
            .stage = vk::ShaderStageFlagBits::eRaygenKHR,
            .module = *raygenSM,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::Miss)] = {
            .stage = vk::ShaderStageFlagBits::eMissKHR,
            .module = *rayMissSM,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::ClosestHit)] = {
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
            .module = *closestHitSM,
            .pName = "main",
        };
        _shaderStages[static_cast<uint32_t>(StageIndex::AnyHit)] = {
            .stage = vk::ShaderStageFlagBits::eAnyHitKHR,
            .module = *anyHitSM,
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

    if (raygenSM.has_value())
        _device->logical().destroy(*raygenSM);
    if (rayMissSM.has_value())
        _device->logical().destroy(*rayMissSM);
    if (closestHitSM.has_value())
        _device->logical().destroy(*closestHitSM);
    if (anyHitSM.has_value())
        _device->logical().destroy(*anyHitSM);

    return false;
}

void RTRenderer::createDescriptorSets()
{
    const vk::DescriptorSetLayoutBinding layoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
    };
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &layoutBinding,
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    _resources->staticDescriptorsAlloc.allocate(layouts, _descriptorSets);
}

void RTRenderer::updateDescriptorSets()
{
    const vk::DescriptorImageInfo colorInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    StaticArray<vk::WriteDescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorWrites;
    for (const auto &ds : _descriptorSets)
    {
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &colorInfo,
        });
    }
    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void RTRenderer::createPipeline(
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{

    StaticArray<vk::DescriptorSetLayout, 8> setLayouts{VK_NULL_HANDLE};
    setLayouts[sCameraBindingSet] = camDSLayout;
    setLayouts[sRTBindingSet] = worldDSLayouts.rayTracing;
    setLayouts[sOutputBindingSet] = _descriptorSetLayout;
    setLayouts[sMaterialsBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[sGeometryBindingSet] = worldDSLayouts.geometry;
    setLayouts[sSkyboxBindingSet] = worldDSLayouts.skyboxOnly;
    setLayouts[sModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;
    setLayouts[sLightsBindingSet] = worldDSLayouts.lights;

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
        .byteSize = sbtSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress |
                 vk::BufferUsageFlagBits::eShaderBindingTableKHR,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
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
