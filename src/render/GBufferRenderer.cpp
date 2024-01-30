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
const vk::Format sNormalMetalnessFormat = vk::Format::eR16G16B16A16Sfloat;

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
};

} // namespace

void GBufferRenderer::init(
    ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc, RenderResources *resources,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(resources != nullptr);
    WHEELS_ASSERT(staticDescriptorsAlloc != nullptr);

    _device = device;
    _resources = resources;

    printf("Creating GBufferRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createGraphicsPipelines(camDSLayout, worldDSLayouts);

    _initialized = true;
}

GBufferRenderer::~GBufferRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipeline();

        _device->logical().destroy(_meshSetLayout);

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void GBufferRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_initialized);

    WHEELS_ASSERT(_meshReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_meshReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
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
    const uint32_t nextFrame, SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(meshletCuller != nullptr);
    WHEELS_ASSERT(sceneStats != nullptr);
    WHEELS_ASSERT(profiler != nullptr);

    GBufferRendererOutput ret;
    {
        ret = createOutputs(renderArea.extent);

        const MeshletCullerOutput cullerOutput = meshletCuller->record(
            scopeAlloc.child_scope(), cb, MeshletCuller::Mode::Opaque, world,
            cam, nextFrame, "GBuffer", sceneStats, profiler);

        updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, cullerOutput, inOutDrawStats);

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
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

        const Attachments attachments = createAttachments(ret);

        const auto _s = profiler->createCpuGpuScope(cb, "GBuffer", true);

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount =
                asserted_cast<uint32_t>(attachments.color.size()),
            .pColorAttachments = attachments.color.data(),
            .pDepthAttachment = &attachments.depth,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

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
        descriptorSets[DrawStatsBindingSet] = _meshSets[nextFrame];

        const StaticArray dynamicOffsets{{
            cam.bufferOffset(),
            worldByteOffsets.globalMaterialConstants,
            worldByteOffsets.modelInstanceTransforms,
            worldByteOffsets.previousModelInstanceTransforms,
            worldByteOffsets.modelInstanceScales,
        }};

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        setViewportScissor(cb, renderArea);

        const PCBlock pcBlock{
            .previousTransformValid = scene.previousTransformsValid ? 1u : 0u,
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eMeshEXT,
            0, // offset
            sizeof(PCBlock), &pcBlock);

        const vk::Buffer argumentHandle =
            _resources->buffers.nativeHandle(cullerOutput.argumentBuffer);
        cb.drawMeshTasksIndirectEXT(argumentHandle, 0, 1, 0);

        cb.endRendering();

        _resources->buffers.release(cullerOutput.dataBuffer);
        _resources->buffers.release(cullerOutput.argumentBuffer);
    }

    return ret;
}

bool GBufferRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    printf("Compiling GBufferRenderer shaders\n");

    const vk::PhysicalDeviceMeshShaderPropertiesEXT &meshShaderProps =
        _device->properties().meshShader;

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
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.mesh",
                                          .debugName = "gbufferMS",
                                          .defines = meshDefines,
                                      });

    const size_t fragDefsLen = 174;
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
    WHEELS_ASSERT(fragDefines.size() <= fragDefsLen);

    Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/gbuffer.frag",
                                          .debugName = "gbuffferPS",
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

void GBufferRenderer::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    WHEELS_ASSERT(_meshReflection.has_value());
    _meshSetLayout = _meshReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), *_device, DrawStatsBindingSet,
        vk::ShaderStageFlagBits::eMeshEXT);

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _meshSetLayout};
    staticDescriptorsAlloc->allocate(layouts, _meshSets);
}

void GBufferRenderer::updateDescriptorSet(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?
    const vk::DescriptorSet ds = _meshSets[nextFrame];

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

void GBufferRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

GBufferRendererOutput GBufferRenderer::createOutputs(
    const vk::Extent2D &size) const
{
    return GBufferRendererOutput{
        .albedoRoughness = _resources->images.create(
            ImageDescription{
                .format = sAlbedoRoughnessFormat,
                .width = size.width,
                .height = size.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eSampled |         // Debug
                    vk::ImageUsageFlagBits::eColorAttachment | // Render
                    vk::ImageUsageFlagBits::eStorage,          // Shading
            },
            "albedoRoughness"),
        .normalMetalness = _resources->images.create(
            ImageDescription{
                .format = sNormalMetalnessFormat,
                .width = size.width,
                .height = size.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eSampled |         // Debug
                    vk::ImageUsageFlagBits::eColorAttachment | // Render
                    vk::ImageUsageFlagBits::eStorage,          // Shading
            },
            "normalMetalness"),
        .velocity = createVelocity(*_resources, size, "velocity"),
        .depth = createDepth(*_device, *_resources, size, "depth"),
    };
}

GBufferRenderer::Attachments GBufferRenderer::createAttachments(
    const GBufferRendererOutput &output) const
{
    return Attachments{
        .color = {{
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(output.albedoRoughness).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(output.normalMetalness).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
            vk::RenderingAttachmentInfo{
                .imageView = _resources->images.resource(output.velocity).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
        }},
        .depth =
            vk::RenderingAttachmentInfo{
                .imageView = _resources->images.resource(output.depth).view,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
            },
    };
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
    setLayouts[DrawStatsBindingSet] = _meshSetLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eMeshEXT,
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

    const StaticArray colorAttachmentFormats{{
        sAlbedoRoughnessFormat,
        sNormalMetalnessFormat,
        sVelocityFormat,
    }};

    const StaticArray<vk::PipelineColorBlendAttachmentState, 3>
        colorBlendAttachments{opaqueColorBlendAttachment()};

    _pipeline = createGraphicsPipeline(
        _device->logical(),
        GraphicsPipelineInfo{
            .layout = _pipelineLayout,
            .colorBlendAttachments = colorBlendAttachments,
            .shaderStages = _shaderStages,
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
