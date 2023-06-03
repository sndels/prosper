#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "LightClustering.hpp"
#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sLightsBindingSet = 0;
constexpr uint32_t sLightClustersBindingSet = 1;
constexpr uint32_t sCameraBindingSet = 2;
constexpr uint32_t sMaterialsBindingSet = 3;
constexpr uint32_t sGeometryBuffersBindingSet = 4;
constexpr uint32_t sModelInstanceTrfnsBindingSet = 5;

struct PCBlock
{
    uint32_t modelInstanceID{0xFFFFFFFF};
    uint32_t meshID{0xFFFFFFFF};
    uint32_t materialID{0xFFFFFFFF};
    uint32_t drawType{0};
};

constexpr std::array<
    const char *, static_cast<size_t>(Renderer::DrawType::Count)>
    sDrawTypeNames = {"Default", DEBUG_DRAW_TYPES_STRS};

vk::Rect2D getRenderArea(
    const RenderResources &resources, const Renderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.illumination).extent;
    assert(targetExtent.depth == 1);
    assert(
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

} // namespace

Renderer::Renderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const InputDSLayouts &dsLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating Renderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
        throw std::runtime_error("Renderer shader compilation failed");

    createGraphicsPipelines(dsLayouts);
}

Renderer::~Renderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipelines();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void Renderer::recompileShaders(
    ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), dsLayouts.world))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(dsLayouts);
    }
}

void Renderer::drawUi()
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

Renderer::OpaqueOutput Renderer::recordOpaque(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, const LightClustering::Output &lightClusters,
    uint32_t nextFrame, Profiler *profiler)
{
    OpaqueOutput ret;
    ret.illumination =
        createIllumination(*_resources, renderArea.extent, "illumination");
    ret.depth = createDepth(*_device, *_resources, renderArea.extent, "depth");

    record(
        cb, world, cam, nextFrame,
        RecordInOut{
            .illumination = ret.illumination,
            .depth = ret.depth,
        },
        lightClusters, false, profiler, "OpaqueGeometry");

    return ret;
}

void Renderer::recordTransparent(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const RecordInOut &inOutTargets,
    const LightClustering::Output &lightClusters, uint32_t nextFrame,
    Profiler *profiler)
{
    record(
        cb, world, cam, nextFrame, inOutTargets, lightClusters, true, profiler,
        "TransparentGeometry");
}

bool Renderer::compileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    printf("Compiling Renderer shaders\n");

    String vertDefines{scopeAlloc, 128};
    appendDefineStr(vertDefines, "CAMERA_SET", sCameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", sGeometryBuffersBindingSet);
    appendDefineStr(
        vertDefines, "MODEL_INSTANCE_TRFNS_SET", sModelInstanceTrfnsBindingSet);
    const Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/forward.vert",
                                          .debugName = "geometryVS",
                                          .defines = vertDefines,
                                      });

    String fragDefines{scopeAlloc, 256};
    appendDefineStr(fragDefines, "LIGHTS_SET", sLightsBindingSet);
    appendDefineStr(
        fragDefines, "LIGHT_CLUSTERS_SET", sLightClustersBindingSet);
    appendDefineStr(fragDefines, "CAMERA_SET", sCameraBindingSet);
    appendDefineStr(fragDefines, "MATERIALS_SET", sMaterialsBindingSet);
    appendDefineStr(
        fragDefines, "NUM_MATERIAL_SAMPLERS",
        worldDSLayouts.materialSamplerCount);
    appendEnumVariantsAsDefines(
        fragDefines, "DrawType",
        Span{sDrawTypeNames.data(), sDrawTypeNames.size()});
    LightClustering::appendShaderDefines(fragDefines);
    PointLights::appendShaderDefines(fragDefines);
    SpotLights::appendShaderDefines(fragDefines);

    const Optional<Device::ShaderCompileResult> fragResult =
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

        const ShaderReflection &vertReflection = vertResult->reflection;
        assert(sizeof(PCBlock) == vertReflection.pushConstantsBytesize());

        const ShaderReflection &fragReflection = fragResult->reflection;
        assert(sizeof(PCBlock) == fragReflection.pushConstantsBytesize());

        _shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertResult->module,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragResult->module,
                .pName = "main",
            }};

        return true;
    }

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void Renderer::destroyGraphicsPipelines()
{
    for (auto &p : _pipelines)
        _device->logical().destroy(p);
    _device->logical().destroy(_pipelineLayout);
}

void Renderer::createGraphicsPipelines(const InputDSLayouts &dsLayouts)
{
    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // Dynamic state
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const vk::PipelineRasterizationStateCreateInfo rasterizerState{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess,
    };

    const vk::PipelineColorBlendAttachmentState opaqueColorBlendAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo opaqueColorBlendState{
        .attachmentCount = 1,
        .pAttachments = &opaqueColorBlendAttachment,
    };

    const StaticArray dynamicStates = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = asserted_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    StaticArray<vk::DescriptorSetLayout, 6> setLayouts{VK_NULL_HANDLE};
    setLayouts[sLightsBindingSet] = dsLayouts.world.lights;
    setLayouts[sLightClustersBindingSet] = dsLayouts.lightClusters;
    setLayouts[sCameraBindingSet] = dsLayouts.camera;
    setLayouts[sMaterialsBindingSet] = dsLayouts.world.materialTextures;
    setLayouts[sGeometryBuffersBindingSet] = dsLayouts.world.geometry;
    setLayouts[sModelInstanceTrfnsBindingSet] = dsLayouts.world.modelInstances;

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

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo>
        pipelineChain{
            vk::GraphicsPipelineCreateInfo{
                .stageCount = asserted_cast<uint32_t>(_shaderStages.size()),
                .pStages = _shaderStages.data(),
                .pVertexInputState = &vertInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizerState,
                .pMultisampleState = &multisampleState,
                .pDepthStencilState = &depthStencilState,
                .pColorBlendState = &opaqueColorBlendState,
                .pDynamicState = &dynamicState,
                .layout = _pipelineLayout,
            },
            vk::PipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &sIlluminationFormat,
                .depthAttachmentFormat = sDepthFormat,
            }};

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{},
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipelines[0] = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipelines[0])),
                .pObjectName = "Renderer::Opaque",
            });
    }

    const vk::PipelineColorBlendAttachmentState transparentColorBlendAttachment{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo transparentColorBlendState{
        .attachmentCount = 1,
        .pAttachments = &transparentColorBlendAttachment,
    };
    pipelineChain.get<vk::GraphicsPipelineCreateInfo>().pColorBlendState =
        &transparentColorBlendState;
    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{},
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipelines[1] = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipelines[1])),
                .pObjectName = "Renderer::Transparent",
            });
    }
}

void Renderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const uint32_t nextFrame, const RecordInOut &inOutTargets,
    const LightClustering::Output &lightClusters, bool transparents,
    Profiler *profiler, const char *debugName)
{
    const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

    const size_t pipelineIndex = transparents ? 1 : 0;

    recordBarriers(cb, inOutTargets, lightClusters);

    const Attachments attachments =
        createAttachments(inOutTargets, transparents);

    const auto _s = profiler->createCpuGpuScope(cb, debugName);

    cb.beginRendering(vk::RenderingInfo{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachments.color,
        .pDepthAttachment = &attachments.depth,
    });

    cb.bindPipeline(
        vk::PipelineBindPoint::eGraphics, _pipelines[pipelineIndex]);

    const auto &scene = world._scenes[world._currentScene];

    StaticArray<vk::DescriptorSet, 6> descriptorSets{VK_NULL_HANDLE};
    descriptorSets[sLightsBindingSet] = scene.lights.descriptorSets[nextFrame];
    descriptorSets[sLightClustersBindingSet] = lightClusters.descriptorSet;
    descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextFrame);
    descriptorSets[sMaterialsBindingSet] = world._materialTexturesDS;
    descriptorSets[sGeometryBuffersBindingSet] = world._geometryDS;
    descriptorSets[sModelInstanceTrfnsBindingSet] =
        scene.modelInstancesDescriptorSets[nextFrame];

    cb.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        0, nullptr);

    setViewportScissor(cb, renderArea);

    for (const auto &instance : scene.modelInstances)
    {
        const auto &model = world._models[instance.modelID];
        for (const auto &subModel : model.subModels)
        {
            const auto &material = world._materials[subModel.materialID];
            const auto &info = world._meshInfos[subModel.meshID];
            const auto isTransparent =
                material.alphaMode == Material::AlphaMode::Blend;
            if ((transparents && isTransparent) ||
                (!transparents && !isTransparent))
            {
                // TODO: Push buffers and offsets
                const PCBlock pcBlock{
                    .modelInstanceID = instance.id,
                    .meshID = subModel.meshID,
                    .materialID = subModel.materialID,
                    .drawType = static_cast<uint32_t>(_drawType),
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

void Renderer::recordBarriers(
    vk::CommandBuffer cb, const RecordInOut &inOutTargets,
    const LightClustering::Output &lightClusters) const
{
    const StaticArray imageBarriers{
        _resources->images.transitionBarrier(
            inOutTargets.illumination,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
        _resources->images.transitionBarrier(
            inOutTargets.depth,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthAttachmentOptimal,
            }),
        _resources->images.transitionBarrier(
            lightClusters.pointers,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
                .layout = vk::ImageLayout::eGeneral,
            }),
    };

    const StaticArray bufferBarriers{
        _resources->texelBuffers.transitionBarrier(
            lightClusters.indicesCount,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
            }),
        _resources->texelBuffers.transitionBarrier(
            lightClusters.indices,
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

Renderer::Attachments Renderer::createAttachments(
    const RecordInOut &inOutTargets, bool transparents) const
{
    Attachments ret;
    if (transparents)
    {
        ret.color = vk::RenderingAttachmentInfo{
            .imageView =
                _resources->images.resource(inOutTargets.illumination).view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        ret.depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
    }
    else
    {
        ret.color = vk::RenderingAttachmentInfo{
            .imageView =
                _resources->images.resource(inOutTargets.illumination).view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
        };
        ret.depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{std::array{1.f, 0.f, 0.f, 0.f}},
        };
    }

    return ret;
}
