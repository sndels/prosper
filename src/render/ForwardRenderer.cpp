#include "ForwardRenderer.hpp"

#include <imgui.h>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/Material.hpp"
#include "../scene/Mesh.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/SceneStats.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "MeshletCuller.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

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
    GeometryBuffersBindingSet,
    SceneInstancesBindingSet,
    SkyboxBindingSet,
    DrawStatsBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t drawType{0};
    uint32_t ibl{0};
    uint32_t previousTransformValid{0};
};

} // namespace

ForwardRenderer::~ForwardRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipelines();

        _device->logical().destroy(_meshSetLayout);

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void ForwardRenderer::init(
    ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc, RenderResources *resources,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(resources != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    _device = device;
    _resources = resources;

    printf("Creating ForwardRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
        throw std::runtime_error("ForwardRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createGraphicsPipelines(dsLayouts);

    _initialized = true;
}

void ForwardRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(_meshReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_meshReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(dsLayouts);
    }
}

ForwardRenderer::OpaqueOutput ForwardRenderer::recordOpaque(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    MeshletCuller *meshletCuller, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, const LightClusteringOutput &lightClusters,
    BufferHandle inOutDrawStats, uint32_t nextFrame, bool applyIbl,
    DrawType drawType, SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    OpaqueOutput ret;
    ret.illumination =
        createIllumination(*_resources, renderArea.extent, "illumination");
    ret.velocity = createVelocity(*_resources, renderArea.extent, "velocity");
    ret.depth = createDepth(*_device, *_resources, renderArea.extent, "depth");

    record(
        WHEELS_MOV(scopeAlloc), cb, meshletCuller, world, cam, nextFrame,
        RecordInOut{
            .illumination = ret.illumination,
            .velocity = ret.velocity,
            .depth = ret.depth,
        },
        lightClusters, inOutDrawStats,
        Options{
            .ibl = applyIbl,
            .drawType = drawType,
        },
        sceneStats, profiler, "OpaqueGeometry");

    return ret;
}

void ForwardRenderer::recordTransparent(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    MeshletCuller *meshletCuller, const World &world, const Camera &cam,
    const TransparentInOut &inOutTargets,
    const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
    uint32_t nextFrame, DrawType drawType, SceneStats *sceneStats,
    Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    record(
        WHEELS_MOV(scopeAlloc), cb, meshletCuller, world, cam, nextFrame,
        RecordInOut{
            .illumination = inOutTargets.illumination,
            .depth = inOutTargets.depth,
        },
        lightClusters, inOutDrawStats,
        Options{
            .transparents = true,
            .drawType = drawType,
        },
        sceneStats, profiler, "TransparentGeometry");
}

bool ForwardRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    const vk::PhysicalDeviceMeshShaderPropertiesEXT &meshShaderProps =
        _device->properties().meshShader;

    const size_t meshDefsLen = 178;
    String meshDefines{scopeAlloc, meshDefsLen};
    appendDefineStr(meshDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(meshDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        meshDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(meshDefines, "MESH_SHADER_SET", DrawStatsBindingSet);
    appendDefineStr(meshDefines, "MAX_MS_VERTS", sMaxMsVertices);
    appendDefineStr(meshDefines, "MAX_MS_PRIMS", sMaxMsTriangles);
    appendDefineStr(
        meshDefines, "LOCAL_SIZE_X",
        std::min(
            meshShaderProps.maxPreferredMeshWorkGroupInvocations,
            asserted_cast<uint32_t>(sMaxMsTriangles)));
    WHEELS_ASSERT(meshDefines.size() <= meshDefsLen);

    Optional<Device::ShaderCompileResult> meshResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.mesh",
                                          .debugName = "geometryMS",
                                          .defines = meshDefines,
                                      });

    const size_t fragDefsLen = 705;
    String fragDefines{scopeAlloc, fragDefsLen};
    appendDefineStr(fragDefines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(fragDefines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    appendDefineStr(fragDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(fragDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        fragDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(
        fragDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(fragDefines, "SKYBOX_SET", SkyboxBindingSet);
    appendEnumVariantsAsDefines(
        fragDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    appendDefineStr(fragDefines, "USE_MATERIAL_LOD_BIAS");
    LightClustering::appendShaderDefines(fragDefines);
    PointLights::appendShaderDefines(fragDefines);
    SpotLights::appendShaderDefines(fragDefines);
    WHEELS_ASSERT(fragDefines.size() <= fragDefsLen);

    Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.frag",
                                          .debugName = "geometryPS",
                                          .defines = fragDefines,
                                      });

    if (meshResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _meshReflection = WHEELS_MOV(meshResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _meshReflection->pushConstantsBytesize());

        _fragReflection = WHEELS_MOV(fragResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _fragReflection->pushConstantsBytesize());

        _shaderStages = {{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eMeshEXT,
                .module = meshResult->module,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragResult->module,
                .pName = "main",
            },
        }};

        return true;
    }

    if (meshResult.has_value())
        _device->logical().destroy(meshResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void ForwardRenderer::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(_meshReflection.has_value());
    _meshSetLayout = _meshReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), *_device, DrawStatsBindingSet,
        vk::ShaderStageFlagBits::eMeshEXT);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT * 2>
        layouts{_meshSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _meshSets.mut_span());
}

void ForwardRenderer::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame, bool transparents,
    const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds =
        _meshSets[nextFrame * MAX_FRAMES_IN_FLIGHT + (transparents ? 1u : 0u)];

    const StaticArray infos{{
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = _resources->buffers.nativeHandle(inOutDrawStats),
            .range = VK_WHOLE_SIZE,
        }},
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = _resources->buffers.nativeHandle(cullerOutput.dataBuffer),
            .range = VK_WHOLE_SIZE,
        }},
    }};

    WHEELS_ASSERT(_meshReflection.has_value());
    const wheels::Array descriptorWrites =
        _meshReflection->generateDescriptorWrites(
            scopeAlloc, DrawStatsBindingSet, ds, infos);

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void ForwardRenderer::destroyGraphicsPipelines()
{
    for (auto &p : _pipelines)
        _device->logical().destroy(p);
    _device->logical().destroy(_pipelineLayout);
}

void ForwardRenderer::createGraphicsPipelines(const InputDSLayouts &dsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = dsLayouts.world.lights;
    setLayouts[LightClustersBindingSet] = dsLayouts.lightClusters;
    setLayouts[CameraBindingSet] = dsLayouts.camera;
    setLayouts[MaterialDatasBindingSet] = dsLayouts.world.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = dsLayouts.world.materialTextures;
    setLayouts[GeometryBuffersBindingSet] = dsLayouts.world.geometry;
    setLayouts[SceneInstancesBindingSet] = dsLayouts.world.sceneInstances;
    setLayouts[SkyboxBindingSet] = dsLayouts.world.skybox;
    setLayouts[DrawStatsBindingSet] = _meshSetLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eMeshEXT |
                      vk::ShaderStageFlagBits::eFragment,
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

    {
        const StaticArray colorAttachmentFormats{{
            sIlluminationFormat,
            sVelocityFormat,
        }};

        const StaticArray<vk::PipelineColorBlendAttachmentState, 2>
            colorBlendAttachments{opaqueColorBlendAttachment()};

        _pipelines[0] = createGraphicsPipeline(
            _device->logical(),
            GraphicsPipelineInfo{
                .layout = _pipelineLayout,
                .colorBlendAttachments = colorBlendAttachments,
                .shaderStages = _shaderStages,
                .renderingInfo =
                    vk::PipelineRenderingCreateInfo{
                        .colorAttachmentCount = asserted_cast<uint32_t>(
                            colorAttachmentFormats.capacity()),
                        .pColorAttachmentFormats =
                            colorAttachmentFormats.data(),
                        .depthAttachmentFormat = sDepthFormat,
                    },
                .debugName = "ForwardRenderer::Opaque",
            });
    }

    {
        const vk::PipelineColorBlendAttachmentState blendAttachment =
            transparentColorBlendAttachment();

        _pipelines[1] = createGraphicsPipeline(
            _device->logical(),
            GraphicsPipelineInfo{
                .layout = _pipelineLayout,
                .colorBlendAttachments = Span{&blendAttachment, 1},
                .shaderStages = _shaderStages,
                .renderingInfo =
                    vk::PipelineRenderingCreateInfo{
                        .colorAttachmentCount = 1,
                        .pColorAttachmentFormats = &sIlluminationFormat,
                        .depthAttachmentFormat = sDepthFormat,
                    },
                .writeDepth = false,
                .debugName = "ForwardRenderer::Transparent",
            });
    }
}
void ForwardRenderer::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    MeshletCuller *meshletCuller, const World &world, const Camera &cam,
    const uint32_t nextFrame, const RecordInOut &inOutTargets,
    const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
    const Options &options, SceneStats *sceneStats, Profiler *profiler,
    const char *debugName)
{
    WHEELS_ASSERT(meshletCuller != nullptr);
    WHEELS_ASSERT(sceneStats != nullptr);
    WHEELS_ASSERT(profiler != nullptr);

    const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

    const size_t pipelineIndex = options.transparents ? 1 : 0;

    const MeshletCuller::Mode cullerMode =
        options.transparents ? MeshletCuller::Mode::Transparent
                             : MeshletCuller::Mode::Opaque;
    const char *cullerDebugPrefix =
        options.transparents ? "Transparent" : "Opaque";
    const MeshletCullerOutput cullerOutput = meshletCuller->record(
        scopeAlloc.child_scope(), cb, cullerMode, world, cam, nextFrame,
        cullerDebugPrefix, sceneStats, profiler);

    updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame, options.transparents, cullerOutput,
        inOutDrawStats);

    recordBarriers(
        WHEELS_MOV(scopeAlloc), cb, inOutTargets, lightClusters, cullerOutput,
        inOutDrawStats);

    const Attachments attachments =
        createAttachments(inOutTargets, options.transparents);

    const auto _s = profiler->createCpuGpuScope(cb, debugName, true);

    cb.beginRendering(vk::RenderingInfo{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount =
            asserted_cast<uint32_t>(attachments.color.size()),
        .pColorAttachments = attachments.color.data(),
        .pDepthAttachment = &attachments.depth,
    });

    cb.bindPipeline(
        vk::PipelineBindPoint::eGraphics, _pipelines[pipelineIndex]);

    const Scene &scene = world.currentScene();
    const WorldDescriptorSets &worldDSes = world.descriptorSets();
    const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[LightsBindingSet] = worldDSes.lights;
    descriptorSets[LightClustersBindingSet] = lightClusters.descriptorSet;
    descriptorSets[CameraBindingSet] = cam.descriptorSet();
    descriptorSets[MaterialDatasBindingSet] =
        worldDSes.materialDatas[nextFrame];
    descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
    descriptorSets[GeometryBuffersBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[SceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
    descriptorSets[DrawStatsBindingSet] =
        _meshSets[nextFrame * MAX_FRAMES_IN_FLIGHT + pipelineIndex];

    const StaticArray dynamicOffsets{{
        worldByteOffsets.directionalLight,
        worldByteOffsets.pointLights,
        worldByteOffsets.spotLights,
        cam.bufferOffset(),
        worldByteOffsets.globalMaterialConstants,
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
    }};

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    setViewportScissor(cb, renderArea);

    const PCBlock pcBlock{
        .drawType = static_cast<uint32_t>(options.drawType),
        .ibl = static_cast<uint32_t>(options.ibl),
        .previousTransformValid = scene.previousTransformsValid ? 1u : 0u,
    };
    cb.pushConstants(
        _pipelineLayout,
        vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
        0, // offset
        sizeof(PCBlock), &pcBlock);

    const vk::Buffer argumentHandle =
        _resources->buffers.nativeHandle(cullerOutput.argumentBuffer);
    cb.drawMeshTasksIndirectEXT(argumentHandle, 0, 1, 0);

    cb.endRendering();

    _resources->buffers.release(cullerOutput.dataBuffer);
    _resources->buffers.release(cullerOutput.argumentBuffer);
}

void ForwardRenderer::recordBarriers(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const RecordInOut &inOutTargets, const LightClusteringOutput &lightClusters,
    const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats) const
{
    if (inOutTargets.velocity.isValid())
    {
        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {inOutTargets.illumination,
                     ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.velocity,
                     ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
                    {lightClusters.pointers, ImageState::FragmentShaderRead},
                }},
                .buffers = StaticArray<BufferTransition, 3>{{
                    {inOutDrawStats, BufferState::MeshShaderReadWrite},
                    {cullerOutput.dataBuffer, BufferState::MeshShaderRead},
                    {cullerOutput.argumentBuffer,
                     BufferState::DrawIndirectRead},
                }},
                .texelBuffers = StaticArray<TexelBufferTransition, 2>{{
                    {lightClusters.indicesCount,
                     BufferState::FragmentShaderRead},
                    {lightClusters.indices, BufferState::FragmentShaderRead},
                }},
            });
    }
    else
    {
        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 3>{{
                    {inOutTargets.illumination,
                     ImageState::ColorAttachmentReadWrite},
                    {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
                    {lightClusters.pointers, ImageState::FragmentShaderRead},
                }},
                .buffers = StaticArray<BufferTransition, 3>{{
                    {inOutDrawStats, BufferState::MeshShaderReadWrite},
                    {cullerOutput.dataBuffer, BufferState::MeshShaderRead},
                    {cullerOutput.argumentBuffer,
                     BufferState::DrawIndirectRead},
                }},
                .texelBuffers = StaticArray<TexelBufferTransition, 2>{{
                    {lightClusters.indicesCount,
                     BufferState::FragmentShaderRead},
                    {lightClusters.indices, BufferState::FragmentShaderRead},
                }},
            });
    }
}

ForwardRenderer::Attachments ForwardRenderer::createAttachments(
    const RecordInOut &inOutTargets, bool transparents) const
{
    Attachments ret;
    if (transparents)
    {
        ret.color.push_back(vk::RenderingAttachmentInfo{
            .imageView =
                _resources->images.resource(inOutTargets.illumination).view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        });
        ret.depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
    }
    else
    {
        ret.color = {
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.illumination).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.velocity).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
        };
        ret.depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
        };
    }

    return ret;
}

vk::Rect2D ForwardRenderer::getRenderArea(
    const RenderResources &resources,
    const ForwardRenderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.illumination).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);
    WHEELS_ASSERT(
        targetExtent == resources.images.resource(inOutTargets.depth).extent);

    return vk::Rect2D{
        .offset = {0, 0},
        .extent =
            {
                targetExtent.width,
                targetExtent.height,
            },
    };
}
