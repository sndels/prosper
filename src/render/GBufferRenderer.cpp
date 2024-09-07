#include "GBufferRenderer.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/VkUtils.hpp"
#include "render/DrawStats.hpp"
#include "render/MeshletCuller.hpp"
#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "scene/Camera.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"
#include "wheels/assert.hpp"

#include <cstdint>

using namespace glm;
using namespace wheels;

namespace
{

const vk::Format sAlbedoRoughnessFormat = vk::Format::eR8G8B8A8Unorm;
const vk::Format sNormalMetalnessFormat = vk::Format::eA2B10G10R10UnormPack32;

enum BindingSet : uint32_t
{
    CameraBindingSet,
    MaterialDatasBindingSet,
    MaterialTexturesBindingSet,
    GeometryBuffersBindingSet,
    SceneInstancesBindingSet,
    MeshShaderBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t previousTransformValid{0};
    uint32_t drawType{0};
};

struct Attachments
{
    wheels::StaticArray<vk::RenderingAttachmentInfo, 3> color;
    vk::RenderingAttachmentInfo depth;
};

} // namespace

void GBufferRenderer::init(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts, MeshletCuller *meshletCuller,
    HierarchicalDepthDownsampler *hierarchicalDepthDownsampler)
{
    WHEELS_ASSERT(!m_initialized);
    WHEELS_ASSERT(meshletCuller != nullptr);
    WHEELS_ASSERT(hierarchicalDepthDownsampler != nullptr);

    LOG_INFO("Creating GBufferRenderer");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope());
    createGraphicsPipelines(camDSLayout, worldDSLayouts);

    m_meshletCuller = meshletCuller;
    m_hierarchicalDepthDownsampler = hierarchicalDepthDownsampler;

    m_initialized = true;
}

GBufferRenderer::~GBufferRenderer()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    destroyGraphicsPipeline();

    gDevice.logical().destroy(m_meshSetLayout);

    for (auto const &stage : m_shaderStages)
        gDevice.logical().destroyShaderModule(stage.module);
}

void GBufferRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_meshReflection.has_value());
    WHEELS_ASSERT(m_fragReflection.has_value());
    if (!m_meshReflection->affected(changedFiles) &&
        !m_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyGraphicsPipeline();
        createGraphicsPipelines(camDSLayout, worldDSLayouts);
    }
}

GBufferRendererOutput GBufferRenderer::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Rect2D &renderArea,
    BufferHandle inOutDrawStats, DrawType drawType, const uint32_t nextFrame,
    DrawStats *drawStats)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(drawStats != nullptr);

    PROFILER_CPU_GPU_SCOPE(cb, "GBuffer");

    GBufferRendererOutput ret;
    {
        ret = GBufferRendererOutput{
            .albedoRoughness = gRenderResources.images->create(
                ImageDescription{
                    .format = sAlbedoRoughnessFormat,
                    .width = renderArea.extent.width,
                    .height = renderArea.extent.height,
                    .usageFlags =
                        vk::ImageUsageFlagBits::eSampled |         // Debug
                        vk::ImageUsageFlagBits::eColorAttachment | // Render
                        vk::ImageUsageFlagBits::eStorage,          // Shading
                },
                "albedoRoughness"),
            .normalMetalness = gRenderResources.images->create(
                ImageDescription{
                    .format = sNormalMetalnessFormat,
                    .width = renderArea.extent.width,
                    .height = renderArea.extent.height,
                    .usageFlags =
                        vk::ImageUsageFlagBits::eSampled |         // Debug
                        vk::ImageUsageFlagBits::eColorAttachment | // Render
                        vk::ImageUsageFlagBits::eStorage,          // Shading
                },
                "normalMetalness"),
            .velocity = createVelocity(renderArea.extent, "velocity"),
            .depth = createDepth(renderArea.extent, "depth"),
        };

        Optional<ImageHandle> prevHierarchicalDepth;
        if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
            prevHierarchicalDepth = m_previousHierarchicalDepth;

        // Conservative two-phase culling from GPU-Driven Rendering Pipelines
        // by Sebastian Aaltonen

        // First phase:
        // Cull with previous frame hierarchical depth and draw. Store a second
        // draw list with potential culling false positives: all meshlets that
        // were culled based on depth.
        const MeshletCullerFirstPhaseOutput firstPhaseCullingOutput =
            m_meshletCuller->recordFirstPhase(
                scopeAlloc.child_scope(), cb, MeshletCuller::Mode::Opaque,
                world, cam, nextFrame, prevHierarchicalDepth, "GBuffer",
                drawStats);

        if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
            gRenderResources.images->release(m_previousHierarchicalDepth);

        {
            recordDraw(
                scopeAlloc.child_scope(), cb, world, cam, renderArea, nextFrame,
                RecordInOut{
                    .inDataBuffer = firstPhaseCullingOutput.dataBuffer,
                    .inArgumentBuffer = firstPhaseCullingOutput.argumentBuffer,
                    .inOutDrawStats = inOutDrawStats,
                    .outAlbedoRoughness = ret.albedoRoughness,
                    .outNormalMetalness = ret.normalMetalness,
                    .outVelocity = ret.velocity,
                    .outDepth = ret.depth,
                },
                drawType, false, drawStats);

            gRenderResources.buffers->release(
                firstPhaseCullingOutput.dataBuffer);
            gRenderResources.buffers->release(
                firstPhaseCullingOutput.argumentBuffer);
        }

        if (firstPhaseCullingOutput.secondPhaseInput.has_value())
        {
            // Second phase:
            // Another pass over the meshelets that got culled by depth in the
            // first pass, now with hierarchical depth built from the first pass
            // result. This way we'll now draw any meshlets that got disoccluded
            // in the curret frame.
            const ImageHandle currentHierarchicalDepth =
                m_hierarchicalDepthDownsampler->record(
                    scopeAlloc.child_scope(), cb, ret.depth, nextFrame,
                    "GBufferFirstPhase");

            const MeshletCullerSecondPhaseOutput secondPhaseCullingOutput =
                m_meshletCuller->recordSecondPhase(
                    scopeAlloc.child_scope(), cb, world, cam, nextFrame,
                    *firstPhaseCullingOutput.secondPhaseInput,
                    currentHierarchicalDepth, "GBuffer");

            gRenderResources.images->release(currentHierarchicalDepth);
            gRenderResources.buffers->release(
                *firstPhaseCullingOutput.secondPhaseInput);

            recordDraw(
                scopeAlloc.child_scope(), cb, world, cam, renderArea, nextFrame,
                RecordInOut{
                    .inDataBuffer = secondPhaseCullingOutput.dataBuffer,
                    .inArgumentBuffer = secondPhaseCullingOutput.argumentBuffer,
                    .inOutDrawStats = inOutDrawStats,
                    .outAlbedoRoughness = ret.albedoRoughness,
                    .outNormalMetalness = ret.normalMetalness,
                    .outVelocity = ret.velocity,
                    .outDepth = ret.depth,
                },
                drawType, true, drawStats);

            gRenderResources.buffers->release(
                secondPhaseCullingOutput.dataBuffer);
            gRenderResources.buffers->release(
                secondPhaseCullingOutput.argumentBuffer);
        }

        // Potential previous pyramid was already freed during first phase
        m_previousHierarchicalDepth = m_hierarchicalDepthDownsampler->record(
            scopeAlloc.child_scope(), cb, ret.depth, nextFrame,
            "GBufferSecondPhase");
        gRenderResources.images->preserve(m_previousHierarchicalDepth);
    }

    return ret;
}

void GBufferRenderer::releasePreserved()
{
    if (gRenderResources.images->isValidHandle(m_previousHierarchicalDepth))
        gRenderResources.images->release(m_previousHierarchicalDepth);
}

bool GBufferRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    const vk::PhysicalDeviceMeshShaderPropertiesEXT &meshShaderProps =
        gDevice.properties().meshShader;

    const size_t meshDefsLen = 201;
    String meshDefines{scopeAlloc, meshDefsLen};
    appendDefineStr(meshDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(meshDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        meshDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(meshDefines, "MESH_SHADER_SET", MeshShaderBindingSet);
    appendDefineStr(meshDefines, "USE_GBUFFER_PC");
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
                                          .debugName = "gbufferMS",
                                          .defines = meshDefines,
                                      });

    const size_t fragDefsLen = 491;
    String fragDefines{scopeAlloc, fragDefsLen};
    appendDefineStr(fragDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(fragDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        fragDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendDefineStr(
        fragDefines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(fragDefines, "USE_MATERIAL_LOD_BIAS");
    appendEnumVariantsAsDefines(
        fragDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    WHEELS_ASSERT(fragDefines.size() <= fragDefsLen);

    Optional<Device::ShaderCompileResult> fragResult =
        gDevice.compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/gbuffer.frag",
                                          .debugName = "gbuffferPS",
                                          .defines = fragDefines,
                                      });

    if (meshResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : m_shaderStages)
            gDevice.logical().destroyShaderModule(stage.module);

        m_meshReflection = WHEELS_MOV(meshResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == m_meshReflection->pushConstantsBytesize());

        m_fragReflection = WHEELS_MOV(fragResult->reflection);

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

void GBufferRenderer::createDescriptorSets(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_meshReflection.has_value());
    m_meshSetLayout = m_meshReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), MeshShaderBindingSet,
        vk::ShaderStageFlagBits::eMeshEXT);

    const StaticArray<vk::DescriptorSetLayout, sDescriptorSetCount> layouts{
        m_meshSetLayout};
    const StaticArray<const char *, sDescriptorSetCount> debugNames{
        "GBufferMesh"};
    gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_meshSets.mut_span());
}

void GBufferRenderer::updateDescriptorSet(
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
            scopeAlloc, MeshShaderBindingSet, ds, infos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void GBufferRenderer::destroyGraphicsPipeline()
{
    gDevice.logical().destroy(m_pipeline);
    gDevice.logical().destroy(m_pipelineLayout);
}

void GBufferRenderer::createGraphicsPipelines(
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[MaterialDatasBindingSet] = worldDSLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[GeometryBuffersBindingSet] = worldDSLayouts.geometry;
    setLayouts[SceneInstancesBindingSet] = worldDSLayouts.sceneInstances;
    setLayouts[MeshShaderBindingSet] = m_meshSetLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eMeshEXT |
                      vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PCBlock),
    };
    m_pipelineLayout =
        gDevice.logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const StaticArray colorAttachmentFormats{{
        sAlbedoRoughnessFormat,
        sNormalMetalnessFormat,
        sVelocityFormat,
    }};

    const StaticArray<vk::PipelineColorBlendAttachmentState, 3>
        colorBlendAttachments{opaqueColorBlendAttachment()};

    m_pipeline = createGraphicsPipeline(
        gDevice.logical(),
        GraphicsPipelineInfo{
            .layout = m_pipelineLayout,
            .colorBlendAttachments = colorBlendAttachments,
            .shaderStages = m_shaderStages,
            .renderingInfo =
                vk::PipelineRenderingCreateInfo{
                    .colorAttachmentCount = asserted_cast<uint32_t>(
                        colorAttachmentFormats.capacity()),
                    .pColorAttachmentFormats = colorAttachmentFormats.data(),
                    .depthAttachmentFormat = sDepthFormat,
                },
            .debugName = "GBufferRenderer",
        });
}

void GBufferRenderer::recordDraw(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Rect2D &renderArea, uint32_t nextFrame,
    const RecordInOut &inputsOutputs, DrawType drawType, bool isSecondPhase,
    DrawStats *drawStats)
{
    WHEELS_ASSERT(drawStats != nullptr);

    const vk::DescriptorSet ds =
        m_meshSets[nextFrame * 2 + (isSecondPhase ? 1u : 0u)];

    const char *debugName = isSecondPhase ? "  SecondPhase" : "  FirstPhase";

    updateDescriptorSet(
        scopeAlloc.child_scope(), ds,
        DescriptorSetBuffers{
            .dataBuffer = inputsOutputs.inDataBuffer,
            .drawStats = inputsOutputs.inOutDrawStats,
        });

    const ImageState colorAttachmentState =
        isSecondPhase ? ImageState::ColorAttachmentReadWrite
                      : ImageState::ColorAttachmentWrite;

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 4>{{
                {inputsOutputs.outAlbedoRoughness, colorAttachmentState},
                {inputsOutputs.outNormalMetalness, colorAttachmentState},
                {inputsOutputs.outVelocity, colorAttachmentState},
                {inputsOutputs.outDepth, ImageState::DepthAttachmentReadWrite},
            }},
            .buffers = StaticArray<BufferTransition, 3>{{
                {inputsOutputs.inOutDrawStats,
                 BufferState::MeshShaderReadWrite},
                {inputsOutputs.inDataBuffer, BufferState::MeshShaderRead},
                {inputsOutputs.inArgumentBuffer, BufferState::DrawIndirectRead},
            }},
        });

    const vk::AttachmentLoadOp loadOp = isSecondPhase
                                            ? vk::AttachmentLoadOp::eLoad
                                            : vk::AttachmentLoadOp::eClear;
    const Attachments attachments{
        .color = {{
            vk::RenderingAttachmentInfo{
                .imageView = gRenderResources.images
                                 ->resource(inputsOutputs.outAlbedoRoughness)
                                 .view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = loadOp,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
            vk::RenderingAttachmentInfo{
                .imageView = gRenderResources.images
                                 ->resource(inputsOutputs.outNormalMetalness)
                                 .view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = loadOp,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
            vk::RenderingAttachmentInfo{
                .imageView =
                    gRenderResources.images->resource(inputsOutputs.outVelocity)
                        .view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = loadOp,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
        }},
        .depth =
            vk::RenderingAttachmentInfo{
                .imageView =
                    gRenderResources.images->resource(inputsOutputs.outDepth)
                        .view,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = loadOp,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
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

    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

    const Scene &scene = world.currentScene();
    const WorldDescriptorSets &worldDSes = world.descriptorSets();
    const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[CameraBindingSet] = cam.descriptorSet();
    descriptorSets[MaterialDatasBindingSet] =
        worldDSes.materialDatas[nextFrame];
    descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
    descriptorSets[GeometryBuffersBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[SceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[MeshShaderBindingSet] = ds;

    const StaticArray dynamicOffsets{{
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

    const PCBlock pcBlock{
        .previousTransformValid = scene.previousTransformsValid ? 1u : 0u,
        .drawType = static_cast<uint32_t>(drawType),
    };
    cb.pushConstants(
        m_pipelineLayout,
        vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
        0, // offset
        sizeof(PCBlock), &pcBlock);

    const vk::Buffer argumentHandle =
        gRenderResources.buffers->nativeHandle(inputsOutputs.inArgumentBuffer);
    cb.drawMeshTasksIndirectEXT(argumentHandle, 0, 1, 0);

    cb.endRendering();
}
