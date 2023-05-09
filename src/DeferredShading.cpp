#include "DeferredShading.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "LightClustering.hpp"
#include "Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    LightsBindingSet = 0,
    LightClustersBindingSet = 1,
    CameraBindingSet = 2,
    MaterialsBindingSet = 3,
    StorageBindingSet = 4,
    BindingSetCount = 5,
};

struct PCBlock
{
    uint drawType{0};
};

constexpr std::array<
    const char *, static_cast<size_t>(DeferredShading::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

} // namespace

DeferredShading::DeferredShading(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating DeferredShading\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("DeferredShading shader compilation failed");

    const StaticArray layoutBindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.capacity()),
            .pBindings = layoutBindings.data(),
        });

    const vk::SamplerCreateInfo info{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    _depthSampler = _device->logical().createSampler(info);

    recreate(camDSLayout, worldDSLayouts);
}

DeferredShading::~DeferredShading()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
        _device->logical().destroy(_depthSampler);
    }
}

void DeferredShading::recompileShaders(
    wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyPipelines();
        createPipeline(camDSLayout, worldDSLayouts);
    }
}

bool DeferredShading::compileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    printf("Compiling DeferredShading shaders\n");

    String defines{scopeAlloc, 256};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "MATERIALS_SET", MaterialsBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    appendEnumVariantsAsDefines(
        defines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    LightClustering::appendShaderDefines(defines);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);

    const auto compSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/deferred_shading.comp",
                                      .debugName = "DeferredShadingCS",
                                      .defines = defines,
                                  });

    if (compSM.has_value())
    {
        _device->logical().destroy(_compSM);

        _compSM = *compSM;

        return true;
    }

    return false;
}

void DeferredShading::recreate(
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();
    createDescriptorSets();
    createPipeline(camDSLayout, worldDSLayouts);
}

void DeferredShading::drawUi()
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 300.f}, ImGuiCond_Appearing);
    ImGui::Begin(
        "Deferred shading settings", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize);

    auto *currentType = reinterpret_cast<uint32_t *>(&_drawType);
    if (ImGui::BeginCombo("Draw type", sDrawTypeNames[*currentType]))
    {
        for (auto i = 0u; i < static_cast<uint32_t>(DrawType::Count); ++i)
        {
            bool selected = *currentType == i;
            if (ImGui::Selectable(sDrawTypeNames[i], &selected))
                _drawType = static_cast<DrawType>(i);
        }
        ImGui::EndCombo();
    }

    ImGui::End();
}

void DeferredShading::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const uint32_t nextImage, Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const auto _s = profiler->createCpuGpuScope(cb, "DeferredShading");

        const StaticArray barriers{
            _resources->images.albedoRoughness.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
            _resources->images.normalMetalness.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
            _resources->images.sceneDepth.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
            _resources->images.sceneColor.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
        };

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] =
            scene.lights.descriptorSets[nextImage];
        descriptorSets[LightClustersBindingSet] =
            _resources->buffers.lightClusters.descriptorSets[nextImage];
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextImage);
        descriptorSets[MaterialsBindingSet] = world._materialTexturesDS;
        descriptorSets[StorageBindingSet] = _descriptorSets[nextImage];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.capacity()),
            descriptorSets.data(), 0, nullptr);

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
            sizeof(PCBlock), &pcBlock);

        const auto &extent = _resources->images.sceneColor.extent;
        const auto groups =
            (glm::uvec2{extent.width, extent.height} - 1u) / 16u + 1u;
        cb.dispatch(groups.x, groups.y, 1);
    }
}

void DeferredShading::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        // Descriptor sets are cleaned up when the pool is destroyed
    }
}

void DeferredShading::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DeferredShading::createDescriptorSets()
{
    StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    _resources->descriptorAllocator.allocate(layouts, _descriptorSets);

    const vk::DescriptorImageInfo albedoRoughnessInfo{
        .imageView = _resources->images.albedoRoughness.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo normalMetalnessInfo{
        .imageView = _resources->images.normalMetalness.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo sceneColorInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthSamplerInfo{
        .sampler = _depthSampler,
    };
    StaticArray<vk::WriteDescriptorSet, MAX_FRAMES_IN_FLIGHT * 5>
        descriptorWrites;
    for (const auto &ds : _descriptorSets)
    {
        // TODO:
        // Can these be one write as the range is contiguous and the type
        // shared?
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &albedoRoughnessInfo,
        });
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &normalMetalnessInfo,
        });
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &depthInfo,
        });
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &sceneColorInfo,
        });
        descriptorWrites.push_back({
            .dstSet = ds,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &depthSamplerInfo,
        });
    }
    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DeferredShading::createPipeline(
    vk::DescriptorSetLayout camDSLayout, const World::DSLayouts &worldDSLayouts)
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = worldDSLayouts.lights;
    setLayouts[LightClustersBindingSet] =
        _resources->buffers.lightClusters.descriptorSetLayout;
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[MaterialsBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[StorageBindingSet] = _descriptorSetLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.capacity()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::ComputePipelineCreateInfo createInfo{
        .stage =
            {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = _compSM,
                .pName = "main",
            },
        .layout = _pipelineLayout,
    };

    {
        auto pipeline = _device->logical().createComputePipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "DeferredShading",
            });
    }
}
