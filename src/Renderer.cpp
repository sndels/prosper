#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "LightClustering.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

const vk::Format sDepthFormat = vk::Format::eD32Sfloat;

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

} // namespace

Renderer::Renderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const vk::Extent2D &renderExtent, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating Renderer\n");

    if (!compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
        throw std::runtime_error("Renderer shader compilation failed");

    recreate(renderExtent, camDSLayout, worldDSLayouts);
}

Renderer::~Renderer()
{
    if (_device != nullptr)
    {
        destroyViewportRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void Renderer::recompileShaders(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope(), worldDSLayouts))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(camDSLayout, worldDSLayouts);
    }
}

void Renderer::recreate(
    const vk::Extent2D &renderExtent, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroyViewportRelated();

    createOutputs(renderExtent);
    createAttachments();
    createGraphicsPipelines(camDSLayout, worldDSLayouts);
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

void Renderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const vk::Rect2D &renderArea, const uint32_t nextFrame,
    bool render_transparents, Profiler *profiler) const
{
    assert(profiler != nullptr);

    const auto pipelineIndex = render_transparents ? 1 : 0;
    {
        const auto _s = profiler->createCpuGpuScope(
            cb, render_transparents ? "Transparent" : "Opaque");

        const StaticArray imageBarriers{
            _resources->staticImages.sceneColor.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
            _resources->staticImages.sceneDepth.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthAttachmentOptimal,
            }),
            _resources->staticBuffers.lightClusters.pointers.transitionBarrier(
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                    .layout = vk::ImageLayout::eGeneral,
                }),
        };

        const StaticArray bufferBarriers{
            _resources->staticBuffers.lightClusters.indicesCount
                .transitionBarrier(BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead,
                }),
            _resources->staticBuffers.lightClusters.indices.transitionBarrier(
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

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &_colorAttachments[pipelineIndex],
            .pDepthAttachment = &_depthAttachments[pipelineIndex],
        });

        cb.bindPipeline(
            vk::PipelineBindPoint::eGraphics, _pipelines[pipelineIndex]);

        const auto &scene = world._scenes[world._currentScene];

        StaticArray<vk::DescriptorSet, 6> descriptorSets{VK_NULL_HANDLE};
        descriptorSets[sLightsBindingSet] =
            scene.lights.descriptorSets[nextFrame];
        descriptorSets[sLightClustersBindingSet] =
            _resources->staticBuffers.lightClusters.descriptorSets[nextFrame];
        descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextFrame);
        descriptorSets[sMaterialsBindingSet] = world._materialTexturesDS;
        descriptorSets[sGeometryBuffersBindingSet] = world._geometryDS;
        descriptorSets[sModelInstanceTrfnsBindingSet] =
            scene.modelInstancesDescriptorSets[nextFrame];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        const vk::Viewport viewport{
            .x = 0.f,
            .y = 0.f,
            .width = static_cast<float>(renderArea.extent.width),
            .height = static_cast<float>(renderArea.extent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        cb.setViewport(0, 1, &viewport);

        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = renderArea.extent,
        };
        cb.setScissor(0, 1, &scissor);

        for (const auto &instance : scene.modelInstances)
        {
            const auto &model = world._models[instance.modelID];
            for (const auto &subModel : model.subModels)
            {
                const auto &material = world._materials[subModel.materialID];
                const auto &info = world._meshInfos[subModel.meshID];
                const auto isTransparent =
                    material.alphaMode == Material::AlphaMode::Blend;
                if ((render_transparents && isTransparent) ||
                    (!render_transparents && !isTransparent))
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
    const auto vertSM = _device->compileShaderModule(
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

    const auto fragSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/forward.frag",
                                      .debugName = "geometryPS",
                                      .defines = fragDefines,
                                  });

    if (vertSM.has_value() && fragSM.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertSM,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragSM,
                .pName = "main",
            }};

        return true;
    }

    if (vertSM.has_value())
        _device->logical().destroy(*vertSM);
    if (fragSM.has_value())
        _device->logical().destroy(*fragSM);

    return false;
}

void Renderer::destroyViewportRelated()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipelines();

        _device->destroy(_resources->staticImages.sceneColor);
        _device->destroy(_resources->staticImages.sceneDepth);

        _colorAttachments.resize(_colorAttachments.capacity(), {});
        _depthAttachments.resize(_colorAttachments.capacity(), {});
    }
}

void Renderer::destroyGraphicsPipelines()
{
    for (auto &p : _pipelines)
        _device->logical().destroy(p);
    _device->logical().destroy(_pipelineLayout);
}

void Renderer::createOutputs(const vk::Extent2D &renderExtent)
{
    {
        _resources->staticImages.sceneColor =
            _device->createImage(ImageCreateInfo{
                .desc =
                    ImageDescription{
                        .format = vk::Format::eR16G16B16A16Sfloat,
                        .width = renderExtent.width,
                        .height = renderExtent.height,
                        .usageFlags =
                            vk::ImageUsageFlagBits::eColorAttachment | // Render
                            vk::ImageUsageFlagBits::eStorage, // ToneMap
                    },
                .debugName = "sceneColor",
            });
    }
    {
        // Check depth buffer without stencil is supported
        const auto features =
            vk::FormatFeatureFlagBits::eDepthStencilAttachment;
        const auto properties =
            _device->physical().getFormatProperties(sDepthFormat);
        if ((properties.optimalTilingFeatures & features) != features)
            throw std::runtime_error("Depth format unsupported");

        _resources->staticImages
            .sceneDepth = _device->createImage(ImageCreateInfo{
            .desc =
                ImageDescription{
                    .format = sDepthFormat,
                    .width = renderExtent.width,
                    .height = renderExtent.height,
                    .usageFlags =
                        vk::ImageUsageFlagBits::
                            eDepthStencilAttachment |     // Geometry
                        vk::ImageUsageFlagBits::eSampled, // Deferred shading
                },
            .debugName = "sceneDepth",
        });

        const auto commandBuffer = _device->beginGraphicsCommands();

        _resources->staticImages.sceneDepth.transition(
            commandBuffer,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            });

        _device->endGraphicsCommands(commandBuffer);
    }
}

void Renderer::createAttachments()
{
    _colorAttachments[0] = vk::RenderingAttachmentInfo{
        .imageView = _resources->staticImages.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array{0.f, 0.f, 0.f, 0.f}},
    };
    _colorAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = _resources->staticImages.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    _depthAttachments[0] = vk::RenderingAttachmentInfo{
        .imageView = _resources->staticImages.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array{1.f, 0.f, 0.f, 0.f}},
    };
    _depthAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = _resources->staticImages.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

void Renderer::createGraphicsPipelines(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
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
    setLayouts[sLightsBindingSet] = worldDSLayouts.lights;
    setLayouts[sLightClustersBindingSet] =
        _resources->staticBuffers.lightClusters.descriptorSetLayout;
    setLayouts[sCameraBindingSet] = camDSLayout;
    setLayouts[sMaterialsBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[sGeometryBuffersBindingSet] = worldDSLayouts.geometry;
    setLayouts[sModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;

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
                .pColorAttachmentFormats =
                    &_resources->staticImages.sceneColor.format,
                .depthAttachmentFormat =
                    _resources->staticImages.sceneDepth.format,
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
