#include "RTRenderer.hpp"

#include "Utils.hpp"
#include "VkUtils.hpp"

// Based on RT Gems II chapter 16

namespace
{

enum StageIndex : uint32_t
{
    RayGen = 0,
    ClosestHit,
    Miss,
};

}

RTRenderer::RTRenderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating RTRenderer\n");

    if (!compileShaders())
        throw std::runtime_error("RTRenderer shader compilation failed");

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

    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

RTRenderer::~RTRenderer()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();
        _device->logical().destroy(_descriptorSetLayout);
        destroyShaders();
    }
}

void RTRenderer::recompileShaders(
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders())
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
    }
}

void RTRenderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createDescriptorSets(swapConfig);
    createPipeline(camDSLayout, worldDSLayouts);
    createShaderBindingTable();
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer RTRenderer::recordCommandBuffer(
    const World &world, const Camera &cam, const vk::Rect2D &renderArea,
    uint32_t nextImage) const
{
    const auto cb = _commandBuffers[nextImage];
    cb.reset();

    cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    _resources->images.sceneColor.transition(
        cb, ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                .accessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .layout = vk::ImageLayout::eGeneral,
            });

    cb.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = "RT",
    });

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, _pipeline);

    const auto &scene = world._scenes[world._currentScene];

    const std::array<vk::DescriptorSet, 3> descriptorSets{
        cam.descriptorSet(nextImage),
        _descriptorSets[nextImage],
        scene.accelerationStructureDS,
    };
    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eRayTracingKHR, _pipelineLayout, 0,
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        0, nullptr);

    const auto sbtAddr =
        _device->logical().getBufferAddress(vk::BufferDeviceAddressInfo{
            .buffer = _shaderBindingTable.handle,
        });

    const vk::StridedDeviceAddressRegionKHR rayGenRegion{
        .deviceAddress = sbtAddr + _sbtGroupSize * StageIndex::RayGen,
        .stride = _sbtGroupSize,
        .size = _sbtGroupSize,
    };

    const vk::StridedDeviceAddressRegionKHR missRegion{
        .deviceAddress = sbtAddr + _sbtGroupSize * StageIndex::Miss,
        .stride = _sbtGroupSize,
        .size = _sbtGroupSize,
    };

    const vk::StridedDeviceAddressRegionKHR hitRegion{
        .deviceAddress = sbtAddr + _sbtGroupSize * StageIndex::ClosestHit,
        .stride = _sbtGroupSize,
        .size = _sbtGroupSize,
    };

    const vk::StridedDeviceAddressRegionKHR callableRegion;

    assert(renderArea.offset.x == 0 && renderArea.offset.y == 0);
    cb.traceRaysKHR(
        &rayGenRegion, &missRegion, &hitRegion, &callableRegion,
        renderArea.extent.width, renderArea.extent.height, 1);

    cb.endDebugUtilsLabelEXT(); // RT

    cb.end();

    return cb;
}

void RTRenderer::destroyShaders()
{
    for (auto const &stage : _shaderStages)
        _device->logical().destroyShaderModule(stage.module);
}

void RTRenderer::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        if (!_commandBuffers.empty())
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                asserted_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        destroyPipeline();

        _device->destroy(_shaderBindingTable);
    }
}

void RTRenderer::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

bool RTRenderer::compileShaders()
{
    fprintf(stderr, "Compiling RTRenderer shaders\n");

    const auto raygenSM =
        _device->compileShaderModule("shader/rt/scene.rgen", "sceneRGEN");
    const auto rayMissSM =
        _device->compileShaderModule("shader/rt/scene.rmiss", "sceneRMISS");
    const auto closestHitSM =
        _device->compileShaderModule("shader/rt/scene.rchit", "sceneRCHIT");

    if (raygenSM && rayMissSM && closestHitSM)
    {
        destroyShaders();

        _shaderStages[StageIndex::RayGen] = {
            .stage = vk::ShaderStageFlagBits::eRaygenKHR,
            .module = *raygenSM,
            .pName = "main",
        };
        _shaderStages[StageIndex::Miss] = {
            .stage = vk::ShaderStageFlagBits::eMissKHR,
            .module = *rayMissSM,
            .pName = "main",
        };
        _shaderStages[StageIndex::ClosestHit] = {
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
            .module = *closestHitSM,
            .pName = "main",
        };

        _shaderGroups[StageIndex::RayGen] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = StageIndex::RayGen,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        _shaderGroups[StageIndex::Miss] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader = StageIndex::Miss,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        _shaderGroups[StageIndex::ClosestHit] = {
            .type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = StageIndex::ClosestHit,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        return true;
    }

    if (raygenSM)
        _device->logical().destroy(*raygenSM);
    if (rayMissSM)
        _device->logical().destroy(*rayMissSM);
    if (closestHitSM)
        _device->logical().destroy(*closestHitSM);

    return false;
}

void RTRenderer::createDescriptorSets(const SwapchainConfig &swapConfig)
{
    const std::vector<vk::DescriptorSetLayout> layouts(
        swapConfig.imageCount, _descriptorSetLayout);
    _descriptorSets =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _resources->descriptorPools.swapchainRelated,
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        });

    const vk::DescriptorImageInfo colorInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    std::vector<vk::WriteDescriptorSet> descriptorWrites;
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
    const std::array<vk::DescriptorSetLayout, 3> setLayouts{
        camDSLayout,
        _descriptorSetLayout,
        worldDSLayouts.accelerationStructure,
    };
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
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

void RTRenderer::createShaderBindingTable()
{

    const auto groupCount = asserted_cast<uint32_t>(_shaderStages.size());
    const auto groupHandleSize =
        _device->properties().rtPipeline.shaderGroupHandleSize;
    const auto groupBaseAlignment =
        _device->properties().rtPipeline.shaderGroupBaseAlignment;
    _sbtGroupSize =
        (((groupHandleSize - 1) / groupBaseAlignment) + 1) * groupBaseAlignment;

    const auto sbtSize = groupCount * _sbtGroupSize;

    std::vector<uint8_t> shaderHandleStorage(sbtSize);
    checkSuccess(
        _device->logical().getRayTracingShaderGroupHandlesKHR(
            _pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()),
        "getRayTracingShaderGroupHandlesKHR");

    _shaderBindingTable = _device->createBuffer(
        "ShaderBindingTable", sbtSize,
        vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eShaderBindingTableKHR,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        MemoryAccess::HostSequentialWrite);

    void *mapped = _device->map(_shaderBindingTable);

    auto *pData = reinterpret_cast<uint8_t *>(mapped);
    for (auto i = 0u; i < groupCount; ++i)
    {
        memcpy(
            pData, shaderHandleStorage.data() + i * groupHandleSize,
            groupHandleSize);
        pData += _sbtGroupSize;
    }

    _device->unmap(_shaderBindingTable);
}

void RTRenderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount,
        });
}
