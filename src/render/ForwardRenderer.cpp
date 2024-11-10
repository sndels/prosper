#include "ForwardRenderer.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "gfx/VkUtils.hpp"
#include "render/DrawStats.hpp"
#include "render/HierarchicalDepthDownsampler.hpp"
#include "render/LightClustering.hpp"
#include "render/MeshletCuller.hpp"
#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "scene/Camera.hpp"
#include "scene/Light.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

#include <imgui.h>
#include <shader_structs/push_constants/forward.h>

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

struct Attachments
{
    InlineArray<vk::RenderingAttachmentInfo, 2> color;
    vk::RenderingAttachmentInfo depth;
};

} // namespace

ForwardRenderer::~ForwardRenderer()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    destroyGraphicsPipelines();

    gDevice.logical().destroy(m_meshSetLayout);

    for (auto const &stage : m_shaderStages)
        gDevice.logical().destroyShaderModule(stage.module);
}

void ForwardRenderer::init(
    ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts,
    MeshletCuller &meshletCuller,
    HierarchicalDepthDownsampler &hierarchicalDepthDownsampler)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating ForwardRenderer");

    if (!compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
        throw std::runtime_error("ForwardRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope());
    createGraphicsPipelines(dsLayouts);

    m_meshletCuller = &meshletCuller;
    m_hierarchicalDepthDownsampler = &hierarchicalDepthDownsampler;

    m_initialized = true;
}

void ForwardRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_meshReflection.has_value());
    WHEELS_ASSERT(m_fragReflection.has_value());
    if (!m_meshReflection->affected(changedFiles) &&
        !m_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(dsLayouts);
    }
}

void ForwardRenderer::startFrame() { m_nextFrameRecord = 0; }

ForwardRenderer::OpaqueOutput ForwardRenderer::recordOpaque(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Rect2D &renderArea,
    const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
    uint32_t nextFrame, bool applyIbl, DrawType drawType, DrawStats &drawStats)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Opaque");

    OpaqueOutput ret;
    ret.illumination = createIllumination(renderArea.extent, "illumination");
    ret.velocity = createVelocity(renderArea.extent, "velocity");
    ret.depth = createDepth(renderArea.extent, "depth");

    Optional<ImageHandle> prevHierarchicalDepth;
    if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
        prevHierarchicalDepth = m_previousHierarchicalDepth;

    // Conservative two-phase culling from GPU-Driven Rendering Pipelines
    // by Sebastian Aaltonen

    // First phase:
    // Cull with previous frame hierarchical depth and draw. Store a second draw
    // list with potential culling false positives: all meshlets that were
    // culled based on depth.
    const MeshletCullerFirstPhaseOutput firstPhaseCullingOutput =
        m_meshletCuller->recordFirstPhase(
            scopeAlloc.child_scope(), cb, MeshletCuller::Mode::Opaque, world,
            cam, nextFrame, prevHierarchicalDepth, "Opaque", drawStats);

    if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
        gRenderResources.images->release(m_previousHierarchicalDepth);

    recordDraw(
        scopeAlloc.child_scope(), cb, world, cam, nextFrame,
        RecordInOut{
            .inOutIllumination = ret.illumination,
            .inOutVelocity = ret.velocity,
            .inOutDepth = ret.depth,
            .inOutDrawStats = inOutDrawStats,
            .inDataBuffer = firstPhaseCullingOutput.dataBuffer,
            .inArgumentBuffer = firstPhaseCullingOutput.argumentBuffer,
        },
        lightClusters,
        Options{
            .ibl = applyIbl,
            .drawType = drawType,
        },
        "  FirstPhase");

    gRenderResources.buffers->release(firstPhaseCullingOutput.dataBuffer);
    gRenderResources.buffers->release(firstPhaseCullingOutput.argumentBuffer);

    if (firstPhaseCullingOutput.secondPhaseInput.has_value())
    {
        // Second phase:
        // Another pass over the meshelets that got culled by depth in the first
        // pass, now with hierarchical depth built from the first pass result.
        // This way we'll now draw any meshlets that got disoccluded in the
        // curret frame.
        const ImageHandle currentHierarchicalDepth =
            m_hierarchicalDepthDownsampler->record(
                scopeAlloc.child_scope(), cb, ret.depth, nextFrame,
                "OpaqueFirstPhase");

        const MeshletCullerSecondPhaseOutput secondPhaseCullingOutput =
            m_meshletCuller->recordSecondPhase(
                scopeAlloc.child_scope(), cb, world, cam, nextFrame,
                *firstPhaseCullingOutput.secondPhaseInput,
                currentHierarchicalDepth, "Opaque");

        gRenderResources.buffers->release(
            *firstPhaseCullingOutput.secondPhaseInput);
        gRenderResources.images->release(currentHierarchicalDepth);

        recordDraw(
            scopeAlloc.child_scope(), cb, world, cam, nextFrame,
            RecordInOut{
                .inOutIllumination = ret.illumination,
                .inOutVelocity = ret.velocity,
                .inOutDepth = ret.depth,
                .inOutDrawStats = inOutDrawStats,
                .inDataBuffer = secondPhaseCullingOutput.dataBuffer,
                .inArgumentBuffer = secondPhaseCullingOutput.argumentBuffer,
            },
            lightClusters,
            Options{
                .ibl = applyIbl,
                .secondPhase = true,
                .drawType = drawType,
            },
            "  SecondPhase");

        gRenderResources.buffers->release(secondPhaseCullingOutput.dataBuffer);
        gRenderResources.buffers->release(
            secondPhaseCullingOutput.argumentBuffer);
    }

    // Potential previous pyramid was already freed during first phase
    m_previousHierarchicalDepth = m_hierarchicalDepthDownsampler->record(
        WHEELS_MOV(scopeAlloc), cb, ret.depth, nextFrame, "OpaqueSecondPhase");
    gRenderResources.images->preserve(m_previousHierarchicalDepth);

    return ret;
}

void ForwardRenderer::recordTransparent(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const TransparentInOut &inOutTargets,
    const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
    uint32_t nextFrame, DrawType drawType, DrawStats &drawStats)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "Transparent");

    const MeshletCullerFirstPhaseOutput cullerOutput =
        m_meshletCuller->recordFirstPhase(
            scopeAlloc.child_scope(), cb, MeshletCuller::Mode::Transparent,
            world, cam, nextFrame, {}, "Transparent", drawStats);
    WHEELS_ASSERT(!cullerOutput.secondPhaseInput.has_value());

    recordDraw(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        RecordInOut{
            .inOutIllumination = inOutTargets.illumination,
            .inOutDepth = inOutTargets.depth,
            .inOutDrawStats = inOutDrawStats,
            .inDataBuffer = cullerOutput.dataBuffer,
            .inArgumentBuffer = cullerOutput.argumentBuffer,
        },
        lightClusters,
        Options{
            .transparents = true,
            .drawType = drawType,
        },
        "  Geometry");

    gRenderResources.buffers->release(cullerOutput.dataBuffer);
    gRenderResources.buffers->release(cullerOutput.argumentBuffer);
}

void ForwardRenderer::releasePreserved()
{
    if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
        gRenderResources.images->release(m_previousHierarchicalDepth);
}

bool ForwardRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    const vk::PhysicalDeviceMeshShaderPropertiesEXT &meshShaderProps =
        gDevice.properties().meshShader;

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
        gDevice.compileShaderModule(
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
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.frag",
                                          .debugName = "geometryPS",
                                          .defines = fragDefines,
                                      });

    if (meshResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : m_shaderStages)
            gDevice.logical().destroyShaderModule(stage.module);

        m_meshReflection = WHEELS_MOV(meshResult->reflection);
        WHEELS_ASSERT(
            sizeof(ForwardPC) == m_meshReflection->pushConstantsBytesize());

        m_fragReflection = WHEELS_MOV(fragResult->reflection);
        WHEELS_ASSERT(
            sizeof(ForwardPC) == m_fragReflection->pushConstantsBytesize());

        m_shaderStages = {{
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
        gDevice.logical().destroy(meshResult->module);
    if (fragResult.has_value())
        gDevice.logical().destroy(fragResult->module);

    return false;
}

void ForwardRenderer::createDescriptorSets(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_meshReflection.has_value());
    m_meshSetLayout = m_meshReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), DrawStatsBindingSet,
        vk::ShaderStageFlagBits::eMeshEXT);

    const StaticArray<vk::DescriptorSetLayout, sDescriptorSetCount> layouts{
        m_meshSetLayout};
    const StaticArray<const char *, sDescriptorSetCount> debugNames{
        "ForwardMesh"};
    gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_meshSets.mut_span());
}

void ForwardRenderer::updateDescriptorSet(
    ScopedScratch scopeAlloc, vk::DescriptorSet ds,
    const DescriptorSetBuffers &buffers) const
{
    const StaticArray infos{{
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = gRenderResources.buffers->nativeHandle(buffers.drawStats),
            .range = VK_WHOLE_SIZE,
        }},
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer =
                gRenderResources.buffers->nativeHandle(buffers.dataBuffer),
            .range = VK_WHOLE_SIZE,
        }},
    }};

    WHEELS_ASSERT(m_meshReflection.has_value());
    const wheels::Array descriptorWrites =
        m_meshReflection->generateDescriptorWrites(
            scopeAlloc, DrawStatsBindingSet, ds, infos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void ForwardRenderer::destroyGraphicsPipelines()
{
    for (auto &p : m_pipelines)
        gDevice.logical().destroy(p);
    gDevice.logical().destroy(m_pipelineLayout);
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
    setLayouts[DrawStatsBindingSet] = m_meshSetLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eMeshEXT |
                      vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(ForwardPC),
    };
    m_pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
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

        m_pipelines[0] = createGraphicsPipeline(
            gDevice.logical(),
            GraphicsPipelineInfo{
                .layout = m_pipelineLayout,
                .colorBlendAttachments = colorBlendAttachments,
                .shaderStages = m_shaderStages,
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

        m_pipelines[1] = createGraphicsPipeline(
            gDevice.logical(),
            GraphicsPipelineInfo{
                .layout = m_pipelineLayout,
                .colorBlendAttachments = Span{&blendAttachment, 1},
                .shaderStages = m_shaderStages,
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
void ForwardRenderer::recordDraw(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, uint32_t nextFrame, const RecordInOut &inputsOutputs,
    const LightClusteringOutput &lightClusters, const Options &options,
    const char *debugName)
{
    PROFILER_CPU_SCOPE(debugName);

    const vk::Rect2D renderArea = getRect2D(inputsOutputs.inOutIllumination);

    const size_t pipelineIndex = options.transparents ? 1 : 0;

    const uint32_t dsIndex =
        nextFrame * MAX_FRAMES_IN_FLIGHT * 2 + m_nextFrameRecord;
    const vk::DescriptorSet ds = m_meshSets[dsIndex];

    updateDescriptorSet(
        scopeAlloc.child_scope(), ds,
        DescriptorSetBuffers{
            .dataBuffer = inputsOutputs.inDataBuffer,
            .drawStats = inputsOutputs.inOutDrawStats,
        });

    InlineArray<ImageTransition, 4> images;
    images.emplace_back(
        inputsOutputs.inOutIllumination, ImageState::ColorAttachmentReadWrite);
    images.emplace_back(
        inputsOutputs.inOutDepth, ImageState::DepthAttachmentReadWrite);
    images.emplace_back(lightClusters.pointers, ImageState::FragmentShaderRead);
    if (inputsOutputs.inOutVelocity.isValid())
        images.emplace_back(
            inputsOutputs.inOutVelocity, ImageState::ColorAttachmentReadWrite);

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = images,
            .buffers = StaticArray<BufferTransition, 3>{{
                {inputsOutputs.inOutDrawStats,
                 BufferState::MeshShaderReadWrite},
                {inputsOutputs.inDataBuffer, BufferState::MeshShaderRead},
                {inputsOutputs.inArgumentBuffer, BufferState::DrawIndirectRead},
            }},
            .texelBuffers = StaticArray<TexelBufferTransition, 2>{{
                {lightClusters.indicesCount, BufferState::FragmentShaderRead},
                {lightClusters.indices, BufferState::FragmentShaderRead},
            }},
        });

    const vk::AttachmentLoadOp loadOp =
        options.secondPhase || options.transparents
            ? vk::AttachmentLoadOp::eLoad
            : vk::AttachmentLoadOp::eClear;
    Attachments attachments;
    attachments.color.push_back(vk::RenderingAttachmentInfo{
        .imageView =
            gRenderResources.images->resource(inputsOutputs.inOutIllumination)
                .view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = loadOp,
        .storeOp = vk::AttachmentStoreOp::eStore,
    });
    if (!options.transparents)
        attachments.color.push_back(vk::RenderingAttachmentInfo{
            .imageView =
                gRenderResources.images->resource(inputsOutputs.inOutVelocity)
                    .view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = loadOp,
            .storeOp = vk::AttachmentStoreOp::eStore,
        });
    attachments.depth = vk::RenderingAttachmentInfo{
        .imageView =
            gRenderResources.images->resource(inputsOutputs.inOutDepth).view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = loadOp,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };

    PROFILER_GPU_SCOPE_WITH_STATS(cb, debugName);

    cb.beginRendering(vk::RenderingInfo{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount =
            asserted_cast<uint32_t>(attachments.color.size()),
        .pColorAttachments = attachments.color.data(),
        .pDepthAttachment = &attachments.depth,
    });

    cb.bindPipeline(
        vk::PipelineBindPoint::eGraphics, m_pipelines[pipelineIndex]);

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
    descriptorSets[DrawStatsBindingSet] = ds;

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
        vk::PipelineBindPoint::eGraphics, m_pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    setViewportScissor(cb, renderArea);

    const ForwardPC pcBlock{
        .drawType = static_cast<uint32_t>(options.drawType),
        .ibl = static_cast<uint32_t>(options.ibl),
        .previousTransformValid = scene.previousTransformsValid ? 1u : 0u,
    };
    cb.pushConstants(
        m_pipelineLayout,
        vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
        0, // offset
        sizeof(pcBlock), &pcBlock);

    const vk::Buffer argumentHandle =
        gRenderResources.buffers->nativeHandle(inputsOutputs.inArgumentBuffer);
    cb.drawMeshTasksIndirectEXT(argumentHandle, 0, 1, 0);

    cb.endRendering();

    m_nextFrameRecord++;
}
