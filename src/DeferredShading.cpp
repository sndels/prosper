#include "DeferredShading.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "LightClustering.hpp"
#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    LightsBindingSet = 0,
    LightClustersBindingSet = 1,
    CameraBindingSet = 2,
    MaterialDatasBindingSet = 3,
    MaterialTexturesBindingSet = 4,
    StorageBindingSet = 5,
    BindingSetCount = 6,
};

struct PCBlock
{
    uint drawType{0};
};

constexpr std::array<
    const char *, static_cast<size_t>(DeferredShading::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

vk::Extent2D getRenderExtent(
    const RenderResources &resources, const GBufferRenderer::Output &gbuffer)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(gbuffer.albedoRoughness).extent;
    assert(targetExtent.depth == 1);

    return vk::Extent2D{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };
}

} // namespace

DeferredShading::DeferredShading(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const InputDSLayouts &dsLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DeferredShading\n");
    if (!compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
        throw std::runtime_error("DeferredShading shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(dsLayouts);
}

DeferredShading::~DeferredShading()
{
    if (_device != nullptr)
    {
        destroyPipelines();

        _device->logical().destroy(_descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void DeferredShading::recompileShaders(
    wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
    {
        destroyPipelines();
        createPipeline(dsLayouts);
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
    appendDefineStr(defines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        defines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    appendEnumVariantsAsDefines(
        defines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    LightClustering::appendShaderDefines(defines);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/deferred_shading.comp",
                .debugName = "DeferredShadingCS",
                .defines = defines,
            });

    if (compResult.has_value())
    {
        _device->logical().destroy(_compSM);

        ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(PCBlock) == reflection.pushConstantsBytesize());

        _compSM = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

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
    const Input &input, const uint32_t nextFrame, Profiler *profiler)
{
    assert(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getRenderExtent(*_resources, input.gbuffer);

        ret.illumination =
            createIllumination(*_resources, renderExtent, "illumination");

        updateDescriptorSet(
            nextFrame, BoundImages{
                           .albedoRoughness = input.gbuffer.albedoRoughness,
                           .normalMetalness = input.gbuffer.normalMetalness,
                           .depth = input.gbuffer.depth,
                           .illumination = ret.illumination,
                       });

        recordBarriers(cb, input, ret);

        const auto _s = profiler->createCpuGpuScope(cb, "DeferredShading");

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] =
            scene.lights.descriptorSets[nextFrame];
        descriptorSets[LightClustersBindingSet] =
            input.lightClusters.descriptorSet;
        descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[MaterialDatasBindingSet] =
            world._materialDatasDSs[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = world._materialTexturesDS;
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

void DeferredShading::recordBarriers(
    vk::CommandBuffer cb, const Input &input, const Output &output) const
{
    const StaticArray imageBarriers{
        _resources->images.transitionBarrier(
            input.gbuffer.albedoRoughness,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            input.gbuffer.normalMetalness,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            input.gbuffer.depth,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            output.illumination,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            }),
        _resources->images.transitionBarrier(
            input.lightClusters.pointers,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
    };

    const StaticArray bufferBarriers{
        _resources->texelBuffers.transitionBarrier(
            input.lightClusters.indicesCount,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
            }),
        _resources->texelBuffers.transitionBarrier(
            input.lightClusters.indices,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
            }),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });
}

void DeferredShading::destroyPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DeferredShading::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflection->generateLayoutBindings(
            scopeAlloc, StorageBindingSet, vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _descriptorSets);
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
        .sampler = _resources->nearestSampler,
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites<5>(
            StorageBindingSet, ds,
            {
                Pair{0u, DescriptorInfo{albedoRoughnessInfo}},
                Pair{1u, DescriptorInfo{normalMetalnessInfo}},
                Pair{2u, DescriptorInfo{depthInfo}},
                Pair{3u, DescriptorInfo{sceneColorInfo}},
                Pair{4u, DescriptorInfo{depthSamplerInfo}},
            });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void DeferredShading::createPipeline(const InputDSLayouts &dsLayouts)
{
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[LightClustersBindingSet] = dsLayouts.lightClusters;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    setLayouts[MaterialDatasBindingSet] = dsLayouts.world.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = dsLayouts.world.materialTextures;
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

    _pipeline = createComputePipeline(
        _device->logical(), createInfo, "DeferredShading");
}
