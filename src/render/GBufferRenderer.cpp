#include "GBufferRenderer.hpp"

#include <imgui.h>

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
    ModelInstanceTrfnsBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t modelInstanceID{0xFFFFFFFF};
    uint32_t meshID{0xFFFFFFFF};
    uint32_t materialID{0xFFFFFFFF};
    uint32_t previousTransformValid{0};
};

} // namespace

GBufferRenderer::GBufferRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);

    printf("Creating GBufferRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("GBufferRenderer shader compilation failed");

    createGraphicsPipelines(camDSLayout, worldDSLayouts);
}

GBufferRenderer::~GBufferRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipeline();

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
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Rect2D &renderArea, const uint32_t nextFrame,
    SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(sceneStats != nullptr);
    WHEELS_ASSERT(profiler != nullptr);

    GBufferRendererOutput ret;
    {
        ret = createOutputs(renderArea.extent);

        transition(
            WHEELS_MOV(scopeAlloc), *_resources, cb,
            Transitions{
                .images = StaticArray<ImageTransition, 4>{{
                    {ret.albedoRoughness, ImageState::ColorAttachmentWrite},
                    {ret.normalMetalness, ImageState::ColorAttachmentWrite},
                    {ret.velocity, ImageState::ColorAttachmentWrite},
                    {ret.depth, ImageState::DepthAttachmentReadWrite},
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

        const auto &scene = world.currentScene();
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
        descriptorSets[ModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSet;

        const StaticArray dynamicOffsets{{
            cam.bufferOffset(),
            worldByteOffsets.globalMaterialConstants,
            worldByteOffsets.modelInstanceTransforms,
            worldByteOffsets.previousModelInstanceTransforms,
        }};

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            asserted_cast<uint32_t>(dynamicOffsets.size()),
            dynamicOffsets.data());

        setViewportScissor(cb, renderArea);

        const Span<const Model> models = world.models();
        const Span<const Material> materials = world.materials();
        const Span<const MeshInfo> meshInfos = world.meshInfos();
        for (const auto &instance : scene.modelInstances)
        {
            const auto &model = models[instance.modelID];
            for (const auto &subModel : model.subModels)
            {
                const auto &material = materials[subModel.materialID];
                const auto &info = meshInfos[subModel.meshID];
                if (info.indexCount == 0)
                    // Invalid or not yet loaded
                    continue;

                if (material.alphaMode != Material::AlphaMode::Blend)
                {
                    // TODO: Push buffers and offsets
                    const PCBlock pcBlock{
                        .modelInstanceID = instance.id,
                        .meshID = subModel.meshID,
                        .materialID = subModel.materialID,
                        .previousTransformValid =
                            instance.previousTransformValid ? 1u : 0u,
                    };
                    cb.pushConstants(
                        _pipelineLayout,
                        vk::ShaderStageFlagBits::eMeshEXT |
                            vk::ShaderStageFlagBits::eFragment,
                        0, // offset
                        sizeof(PCBlock), &pcBlock);

                    cb.drawMeshTasksEXT(info.meshletCount, 1, 1);

                    sceneStats->totalMeshCount++;
                    sceneStats->totalTriangleCount += info.indexCount / 3;
                    sceneStats->totalMeshletCount += info.meshletCount;
                }
            }
        }

        cb.endRendering();
    }

    return ret;
}

bool GBufferRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    printf("Compiling GBufferRenderer shaders\n");

    const size_t meshDefsLen = 176;
    String meshDefines{scopeAlloc, meshDefsLen};
    appendDefineStr(meshDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(meshDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        meshDefines, "MODEL_INSTANCE_TRFNS_SET", ModelInstanceTrfnsBindingSet);
    appendDefineStr(meshDefines, "USE_GBUFFER_PC");
    appendDefineStr(meshDefines, "MAX_MS_VERTS", sMaxMsVertices);
    appendDefineStr(meshDefines, "MAX_MS_PRIMS", sMaxMsTriangles);
    appendDefineStr(
        meshDefines, "LOCAL_SIZE_X", std::max(sMaxMsVertices, sMaxMsTriangles));
    WHEELS_ASSERT(meshDefines.size() <= meshDefsLen);

    Optional<Device::ShaderCompileResult> meshResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.mesh",
                                          .debugName = "gbufferMS",
                                          .defines = meshDefines,
                                      });

    const size_t fragDefsLen = 150;
    String fragDefines{scopeAlloc, fragDefsLen};
    appendDefineStr(fragDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(fragDefines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        fragDefines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
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
    setLayouts[ModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;

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
