#include "DeferredShading.hpp"

#include <imgui.h>

#include <fstream>

#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "GBufferRenderer.hpp"
#include "LightClustering.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"
#include "Utils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    LightsBindingSet,
    LightClustersBindingSet,
    CameraBindingSet,
    MaterialDatasBindingSet,
    MaterialTexturesBindingSet,
    SkyboxBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint drawType{0};
    uint ibl{0};
};

ComputePass::Shader shaderDefinitionCallback(
    Allocator &alloc, const WorldDSLayouts &worldDSLayouts)
{
    const size_t len = 768;
    String defines{alloc, len};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        defines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    appendDefineStr(defines, "SKYBOX_SET", SkyboxBindingSet);
    appendEnumVariantsAsDefines(
        defines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    LightClustering::appendShaderDefines(defines);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    WHEELS_ASSERT(defines.size() <= len);

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
    setLayouts[SkyboxBindingSet] = dsLayouts.world.skybox;
    return setLayouts;
}

} // namespace

void DeferredShading::init(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(!_initialized);

    _computePass.init(
        WHEELS_MOV(scopeAlloc), staticDescriptorsAlloc,
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
            .externalDsLayouts = externalDsLayouts(dsLayouts),
        });

    _initialized = true;
}

void DeferredShading::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(_initialized);

    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles,
        [&dsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, dsLayouts.world); },
        externalDsLayouts(dsLayouts));
}

DeferredShading::Output DeferredShading::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const Input &input, const uint32_t nextFrame,
    bool applyIbl, DrawType drawType, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getExtent2D(input.gbuffer.albedoRoughness);

        ret.illumination = createIllumination(renderExtent, "illumination");

        _computePass.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.gbuffer.albedoRoughness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.gbuffer.normalMetalness)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.gbuffer.geometryNormal)
                                     .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(input.gbuffer.depth)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 6>{{
                    {input.gbuffer.albedoRoughness,
                     ImageState::ComputeShaderRead},
                    {input.gbuffer.normalMetalness,
                     ImageState::ComputeShaderRead},
                    {input.gbuffer.geometryNormal,
                     ImageState::ComputeShaderRead},
                    {input.gbuffer.depth, ImageState::ComputeShaderRead},
                    {ret.illumination, ImageState::ComputeShaderWrite},
                    {input.lightClusters.pointers,
                     ImageState::ComputeShaderRead},
                }},
                .texelBuffers = StaticArray<TexelBufferTransition, 2>{{
                    {input.lightClusters.indicesCount,
                     BufferState::ComputeShaderRead},
                    {input.lightClusters.indices,
                     BufferState::ComputeShaderRead},
                }},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "DeferredShading");

        const PCBlock pcBlock{
            .drawType = static_cast<uint32_t>(drawType),
            .ibl = static_cast<uint32_t>(applyIbl),
        };

        const WorldDescriptorSets &worldDSes = world.descriptorSets();
        const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[LightsBindingSet] = worldDSes.lights;
        descriptorSets[LightClustersBindingSet] =
            input.lightClusters.descriptorSet;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[MaterialDatasBindingSet] =
            worldDSes.materialDatas[nextFrame];
        descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
        descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
        descriptorSets[StorageBindingSet] = _computePass.storageSet(nextFrame);

        const StaticArray dynamicOffsets{{
            worldByteOffsets.directionalLight,
            worldByteOffsets.pointLights,
            worldByteOffsets.spotLights,
            cam.bufferOffset(),
            worldByteOffsets.globalMaterialConstants,
        }};

        const uvec3 extent =
            glm::uvec3{renderExtent.width, renderExtent.height, 1u};

        _computePass.record(
            cb, pcBlock, extent, descriptorSets, dynamicOffsets);
    }

    return ret;
}
