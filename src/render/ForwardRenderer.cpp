#include "ForwardRenderer.hpp"

#include <imgui.h>

#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/Light.hpp"
#include "../scene/Material.hpp"
#include "../scene/Mesh.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Ui.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "RenderResources.hpp"
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
    GeometryBuffersBindingSet = 5,
    ModelInstanceTrfnsBindingSet = 6,
    SkyboxBindingSet = 7,
    BindingSetCount,
};

struct PCBlock
{
    uint32_t modelInstanceID{0xFFFFFFFF};
    uint32_t meshID{0xFFFFFFFF};
    uint32_t materialID{0xFFFFFFFF};
    uint32_t drawType{0};
    uint32_t ibl{0};
    uint32_t previousTransformValid{0};
};

constexpr StaticArray<
    const char *, static_cast<size_t>(ForwardRenderer::DrawType::Count)>
    sDrawTypeNames{{DEBUG_DRAW_TYPES_STRS}};

} // namespace

ForwardRenderer::ForwardRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const InputDSLayouts &dsLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);

    printf("Creating ForwardRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
        throw std::runtime_error("ForwardRenderer shader compilation failed");

    createGraphicsPipelines(dsLayouts);
}

ForwardRenderer::~ForwardRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipelines();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void ForwardRenderer::recompileShaders(
    ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    const InputDSLayouts &dsLayouts)
{
    WHEELS_ASSERT(_vertReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_vertReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(dsLayouts);
    }
}

void ForwardRenderer::drawUi()
{
    enumDropdown("Draw type", _drawType, sDrawTypeNames);
}

ForwardRenderer::OpaqueOutput ForwardRenderer::recordOpaque(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const vk::Rect2D &renderArea,
    const LightClusteringOutput &lightClusters, uint32_t nextFrame,
    bool applyIbl, Profiler *profiler)
{
    OpaqueOutput ret;
    ret.illumination =
        createIllumination(*_resources, renderArea.extent, "illumination");
    ret.velocity = createVelocity(*_resources, renderArea.extent, "velocity");
    ret.depth = createDepth(*_device, *_resources, renderArea.extent, "depth");

    record(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        RecordInOut{
            .illumination = ret.illumination,
            .velocity = ret.velocity,
            .depth = ret.depth,
        },
        lightClusters, Options{.ibl = applyIbl}, profiler, "OpaqueGeometry");

    return ret;
}

void ForwardRenderer::recordTransparent(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const TransparentInOut &inOutTargets,
    const LightClusteringOutput &lightClusters, uint32_t nextFrame,
    Profiler *profiler)
{
    record(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        RecordInOut{
            .illumination = inOutTargets.illumination,
            .depth = inOutTargets.depth,
        },
        lightClusters, Options{.transparents = true}, profiler,
        "TransparentGeometry");
}

bool ForwardRenderer::compileShaders(
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts)
{
    printf("Compiling ForwardRenderer shaders\n");

    const size_t vertDefsLen = 92;
    String vertDefines{scopeAlloc, vertDefsLen};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);
    appendDefineStr(
        vertDefines, "MODEL_INSTANCE_TRFNS_SET", ModelInstanceTrfnsBindingSet);
    WHEELS_ASSERT(vertDefines.size() <= vertDefsLen);

    Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.vert",
                                          .debugName = "geometryVS",
                                          .defines = vertDefines,
                                      });

    const size_t fragDefsLen = 650;
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

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _vertReflection = WHEELS_MOV(vertResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _vertReflection->pushConstantsBytesize());

        _fragReflection = WHEELS_MOV(fragResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _fragReflection->pushConstantsBytesize());

        _shaderStages = {{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertResult->module,
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

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
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
    setLayouts[ModelInstanceTrfnsBindingSet] = dsLayouts.world.modelInstances;
    setLayouts[SkyboxBindingSet] = dsLayouts.world.skybox;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex |
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

    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

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
                .vertInputInfo = vertInputInfo,
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
                .vertInputInfo = vertInputInfo,
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
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, const uint32_t nextFrame,
    const RecordInOut &inOutTargets, const LightClusteringOutput &lightClusters,
    const Options &options, Profiler *profiler, const char *debugName)
{
    const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

    const size_t pipelineIndex = options.transparents ? 1 : 0;

    recordBarriers(WHEELS_MOV(scopeAlloc), cb, inOutTargets, lightClusters);

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

    const auto &scene = world.currentScene();
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
    descriptorSets[GeometryBuffersBindingSet] = worldDSes.geometry;
    descriptorSets[ModelInstanceTrfnsBindingSet] =
        scene.modelInstancesDescriptorSet;
    descriptorSets[SkyboxBindingSet] = worldDSes.skybox;

    const StaticArray dynamicOffsets{{
        worldByteOffsets.directionalLight,
        worldByteOffsets.pointLights,
        worldByteOffsets.spotLights,
        cam.bufferOffset(),
        worldByteOffsets.globalMaterialConstants,
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
    }};

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        asserted_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

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
            const auto isTransparent =
                material.alphaMode == Material::AlphaMode::Blend;
            if ((options.transparents && isTransparent) ||
                (!options.transparents && !isTransparent))
            {
                // TODO: Push buffers and offsets
                const PCBlock pcBlock{
                    .modelInstanceID = instance.id,
                    .meshID = subModel.meshID,
                    .materialID = subModel.materialID,
                    .drawType = static_cast<uint32_t>(_drawType),
                    .ibl = static_cast<uint32_t>(options.ibl),
                    .previousTransformValid =
                        instance.previousTransformValid ? 1u : 0u,
                };
                cb.pushConstants(
                    _pipelineLayout,
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, // offset
                    sizeof(PCBlock), &pcBlock);

                cb.draw(info.indexCount, 1, 0, 0);
            }
        }
    }

    cb.endRendering();
}

void ForwardRenderer::recordBarriers(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const RecordInOut &inOutTargets,
    const LightClusteringOutput &lightClusters) const
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
