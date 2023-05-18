#include "DeferredShading.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "LightClustering.hpp"
#include "RenderTargets.hpp"
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

    createDescriptorSets();

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

    createPipeline(camDSLayout, worldDSLayouts);
}

DeferredShading::~DeferredShading()
{
    if (_device != nullptr)
    {
        destroyPipelines();

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

void DeferredShading::drawUi()
{
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
}

DeferredShading::Output DeferredShading::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const GBufferRenderer::Output &gbuffer, const uint32_t nextFrame,
    Profiler *profiler)
{
    assert(profiler != nullptr);

    const vk::Extent3D targetExtent =
        _resources->images.resource(gbuffer.albedoRoughness).extent;
    assert(targetExtent.depth == 1);

    const vk::Extent2D renderExtent{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
    assert(
        renderExtent.width ==
        _resources->images.resource(gbuffer.normalMetalness).extent.width);
    assert(
        renderExtent.height ==
        _resources->images.resource(gbuffer.normalMetalness).extent.height);
    assert(
        renderExtent.width ==
        _resources->images.resource(gbuffer.depth).extent.width);
    assert(
        renderExtent.height ==
        _resources->images.resource(gbuffer.depth).extent.height);

    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "DeferredShading");

        ret.illumination =
            createIllumination(*_resources, renderExtent, "illumination");

        updateDescriptorSet(
            nextFrame, BoundImages{
                           .albedoRoughness = gbuffer.albedoRoughness,
                           .normalMetalness = gbuffer.normalMetalness,
                           .depth = gbuffer.depth,
                           .illumination = ret.illumination,
                       });

        const StaticArray barriers{
            _resources->images.transitionBarrier(
                gbuffer.albedoRoughness,
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                    .layout = vk::ImageLayout::eGeneral,
                }),
            _resources->images.transitionBarrier(
                gbuffer.normalMetalness,
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                    .layout = vk::ImageLayout::eGeneral,
                }),
            _resources->images.transitionBarrier(
                gbuffer.depth,
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                    .layout = vk::ImageLayout::eGeneral,
                }),
            _resources->images.transitionBarrier(
                ret.illumination,
                ImageState{
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
            scene.lights.descriptorSets[nextFrame];
        descriptorSets[LightClustersBindingSet] =
            _resources->staticBuffers.lightClusters.descriptorSets[nextFrame];
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[MaterialsBindingSet] = world._materialTexturesDS;
        descriptorSets[StorageBindingSet] = _descriptorSets[nextFrame];

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

        const auto groups =
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
            1u;
        cb.dispatch(groups.x, groups.y, 1);
    }

    return ret;
}

void DeferredShading::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DeferredShading::createDescriptorSets()
{
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

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    _resources->staticDescriptorsAlloc.allocate(layouts, _descriptorSets);
}

void DeferredShading::updateDescriptorSet(
    uint32_t nextFrame, const BoundImages &images)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const vk::DescriptorImageInfo albedoRoughnessInfo{
        .imageView = _resources->images.resource(images.albedoRoughness).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo normalMetalnessInfo{
        .imageView = _resources->images.resource(images.normalMetalness).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthInfo{
        .imageView = _resources->images.resource(images.depth).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo sceneColorInfo{
        .imageView = _resources->images.resource(images.illumination).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    const vk::DescriptorImageInfo depthSamplerInfo{
        .sampler = _depthSampler,
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];
    StaticArray descriptorWrites{
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &albedoRoughnessInfo,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &normalMetalnessInfo,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &depthInfo,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &sceneColorInfo,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &depthSamplerInfo,
        }};
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
        _resources->staticBuffers.lightClusters.descriptorSetLayout;
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
