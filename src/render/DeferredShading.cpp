#include "DeferredShading.hpp"

#include <glm/glm.hpp>
#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "RenderTargets.hpp"

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

ComputePass::Shader shaderDefinitionCallback(
    Allocator &alloc, const World::DSLayouts &worldDSLayouts)
{
    String defines{alloc, 768};
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
    assert(defines.size() <= 768);

    return ComputePass::Shader{
        .relPath = "shader/deferred_shading.comp",
        .debugName = String{alloc, "DeferredShadingCS"},
        .defines = WHEELS_MOV(defines),
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const DeferredShading::InputDSLayouts &dsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[LightClustersBindingSet] = dsLayouts.lightClusters;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    setLayouts[MaterialDatasBindingSet] = dsLayouts.world.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = dsLayouts.world.materialTextures;
    return setLayouts;
}

} // namespace

DeferredShading::DeferredShading(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const InputDSLayouts &dsLayouts)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc),
      device,
      staticDescriptorsAlloc,
      [&dsLayouts](Allocator &alloc)
      { return shaderDefinitionCallback(alloc, dsLayouts.world); },
      StorageBindingSet,
      externalDsLayouts(dsLayouts)}
{
    assert(_resources != nullptr);
}

void DeferredShading::recompileShaders(
    wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc),
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        externalDsLayouts(dsLayouts));
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

        _computePass.updateDescriptorSet(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(input.gbuffer.albedoRoughness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images
                                     .resource(input.gbuffer.normalMetalness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(input.gbuffer.depth).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        _resources->images.resource(ret.illumination).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = _resources->nearestSampler,
                }},
            });

        transition<5, 2>(
            *_resources, cb,
            {
                {input.gbuffer.albedoRoughness, ImageState::ComputeShaderRead},
                {input.gbuffer.normalMetalness, ImageState::ComputeShaderRead},
                {input.gbuffer.depth, ImageState::ComputeShaderRead},
                {ret.illumination, ImageState::ComputeShaderWrite},
                {input.lightClusters.pointers, ImageState::ComputeShaderRead},
            },
            {
                {input.lightClusters.indicesCount,
                 BufferState::ComputeShaderRead},
                {input.lightClusters.indices, BufferState::ComputeShaderRead},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "DeferredShading");

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(_drawType),
        };

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] = world._lightsDescriptorSet;
        descriptorSets[LightClustersBindingSet] =
            input.lightClusters.descriptorSet;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[MaterialDatasBindingSet] =
            world._materialDatasDSs[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = world._materialTexturesDS;
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const StaticArray dynamicOffsets = {
            world._directionalLightByteOffset,
            world._pointLightByteOffset,
            world._spotLightByteOffset,
            cam.bufferOffset(),
        };

        const uvec3 groups = glm::uvec3{
            (glm::uvec2{renderExtent.width, renderExtent.height} - 1u) / 16u +
                1u,
            1u};

        _computePass.record(
            cb, pcBlock, groups, descriptorSets, dynamicOffsets);
    }

    return ret;
}
