#include "GBufferRenderer.hpp"

#include <imgui.h>

#include "../gfx/DescriptorAllocator.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
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

const vk::Format sAlbedoRoughnessFormat = vk::Format::eR8G8B8A8Unorm;
const vk::Format sNormalMetalnessFormat = vk::Format::eA2B10G10R10UnormPack32;

enum BindingSet : uint32_t
{
    CameraBindingSet,
    MaterialDatasBindingSet,
    MaterialTexturesBindingSet,
    GeometryBuffersBindingSet,
    SceneInstancesBindingSet,
    DrawStatsBindingSet,
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
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!m_initialized);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    printf("Creating GBufferRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createGraphicsPipelines(camDSLayout, worldDSLayouts);

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
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    MeshletCuller *meshletCuller, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, BufferHandle inOutDrawStats,
    DrawType drawType, const uint32_t nextFrame, SceneStats *sceneStats)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(meshletCuller != nullptr);
    WHEELS_ASSERT(sceneStats != nullptr);

    PROFILER_CPU_SCOPE("GBuffer");

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

        const MeshletCullerOutput cullerOutput = meshletCuller->record(
            scopeAlloc.child_scope(), cb, MeshletCuller::Mode::Opaque, world,
            cam, nextFrame, "GBuffer", sceneStats);

        updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, cullerOutput, inOutDrawStats);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {ret.albedoRoughness, ImageState::ColorAttachmentWrite},
                    {ret.normalMetalness, ImageState::ColorAttachmentWrite},
                    {ret.velocity, ImageState::ColorAttachmentWrite},
                    {ret.depth, ImageState::DepthAttachmentReadWrite},
                }},
                .buffers = StaticArray<BufferTransition, 3>{{
                    {inOutDrawStats, BufferState::MeshShaderReadWrite},
                    {cullerOutput.dataBuffer, BufferState::MeshShaderRead},
                    {cullerOutput.argumentBuffer,
                     BufferState::DrawIndirectRead},
                }},
            });

        const Attachments attachments{
            .color = {{
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.albedoRoughness)
                            .view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue =
                        vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
                },
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.normalMetalness)
                            .view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue =
                        vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
                },
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.velocity).view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue =
                        vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
                },
            }},
            .depth =
                vk::RenderingAttachmentInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.depth).view,
                    .imageLayout =
                        vk::ImageLayout::eDepthStencilAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue =
                        vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
                },
        };

        PROFILER_GPU_SCOPE_WITH_STATS(cb, "GBuffer");

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
        descriptorSets[GeometryBuffersBindingSet] =
            worldDSes.geometry[nextFrame];
        descriptorSets[SceneInstancesBindingSet] =
            scene.sceneInstancesDescriptorSet;
        descriptorSets[DrawStatsBindingSet] = m_meshSets[nextFrame];

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
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        setViewportScissor(cb, renderArea);

        const PCBlock pcBlock{
            .previousTransformValid = scene.previousTransformsValid ? 1u : 0u,
            .drawType = static_cast<uint32_t>(drawType),
        };
        cb.pushConstants(
            m_pipelineLayout,
            vk::ShaderStageFlagBits::eMeshEXT |
                vk::ShaderStageFlagBits::eFragment,
            0, // offset
            sizeof(PCBlock), &pcBlock);

        const vk::Buffer argumentHandle =
            gRenderResources.buffers->nativeHandle(cullerOutput.argumentBuffer);
        cb.drawMeshTasksIndirectEXT(argumentHandle, 0, 1, 0);

        cb.endRendering();

        gRenderResources.buffers->release(cullerOutput.dataBuffer);
        gRenderResources.buffers->release(cullerOutput.argumentBuffer);
    }

    return ret;
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
    appendDefineStr(meshDefines, "MESH_SHADER_SET", DrawStatsBindingSet);
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

void GBufferRenderer::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(m_meshReflection.has_value());
    m_meshSetLayout = m_meshReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), DrawStatsBindingSet,
        vk::ShaderStageFlagBits::eMeshEXT);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        m_meshSetLayout};
    const StaticArray<const char *, MAX_FRAMES_IN_FLIGHT> debugNames{
        "GBufferMesh"};
    staticDescriptorsAlloc->allocate(
        layouts, debugNames, m_meshSets.mut_span());
}

void GBufferRenderer::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = m_meshSets[nextFrame];

    const StaticArray infos{{
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = gRenderResources.buffers->nativeHandle(inOutDrawStats),
            .range = VK_WHOLE_SIZE,
        }},
        DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer =
                gRenderResources.buffers->nativeHandle(cullerOutput.dataBuffer),
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
    setLayouts[DrawStatsBindingSet] = m_meshSetLayout;

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
