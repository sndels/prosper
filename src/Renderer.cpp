#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

namespace
{

constexpr uint32_t sLightsBindingSet = 0;
constexpr uint32_t sLightClustersBindingSet = 1;
constexpr uint32_t sCameraBindingSet = 2;
constexpr uint32_t sMaterialsBindingSet = 3;
constexpr uint32_t sVertexBuffersBindingSet = 4;
constexpr uint32_t sIndexBuffersBindingSet = 5;
constexpr uint32_t sModelInstanceTrfnsBindingSet = 6;

} // namespace

Renderer::Renderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating Renderer\n");

    if (!compileShaders(worldDSLayouts))
        throw std::runtime_error("Renderer shader compilation failed");

    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

Renderer::~Renderer()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void Renderer::recompileShaders(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(worldDSLayouts))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    }
}

void Renderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createOutputs(swapConfig);
    createAttachments();
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer Renderer::recordCommandBuffer(
    const World &world, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage, bool render_transparents) const
{
    const auto pipelineIndex = render_transparents ? 1 : 0;
    // Separate buffers for opaque and transparent
    // opaque uses 0, 2,... and transparent 1,3,...
    const auto buffer = _commandBuffers[nextImage * 2 + pipelineIndex];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    const std::array<vk::ImageMemoryBarrier2, 3> imageBarriers{
        _resources->images.sceneColor.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        }),
        _resources->images.sceneDepth.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .layout = vk::ImageLayout::eDepthAttachmentOptimal,
        }),
        _resources->buffers.lightClusters.pointers.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .accessMask = vk::AccessFlagBits2::eShaderRead,
            .layout = vk::ImageLayout::eGeneral,
        }),
    };

    const std::array<vk::BufferMemoryBarrier2, 2> bufferBarriers{
        _resources->buffers.lightClusters.indicesCount.transitionBarrier(
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderRead,
            }),
        _resources->buffers.lightClusters.indices.transitionBarrier(BufferState{
            .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .accessMask = vk::AccessFlagBits2::eShaderRead,
        }),
    };

    buffer.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount =
            asserted_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });

    buffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = render_transparents ? "Transparent" : "Opaque",
    });

    buffer.beginRendering(vk::RenderingInfo{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &_colorAttachments[pipelineIndex],
        .pDepthAttachment = &_depthAttachments[pipelineIndex],
    });

    buffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics, _pipelines[pipelineIndex]);

    const auto &scene = world._scenes[world._currentScene];

    std::array<vk::DescriptorSet, 7> descriptorSets = {};
    descriptorSets[sLightsBindingSet] = scene.lights.descriptorSets[nextImage];
    descriptorSets[sLightClustersBindingSet] =
        _resources->buffers.lightClusters.descriptorSets[nextImage];
    descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextImage);
    descriptorSets[sMaterialsBindingSet] = world._materialTexturesDS;
    descriptorSets[sVertexBuffersBindingSet] = world._vertexBuffersDS;
    descriptorSets[sIndexBuffersBindingSet] = world._indexBuffersDS;
    descriptorSets[sModelInstanceTrfnsBindingSet] =
        scene.modelInstancesDescriptorSets[nextImage];

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        0, nullptr);

    for (const auto &instance : scene.modelInstances)
    {
        const auto &model = world._models[instance.modelID];
        for (const auto &subModel : model.subModels)
        {
            const auto &material = world._materials[subModel.materialID];
            const auto &mesh = world._meshes[subModel.meshID];
            const auto isTransparent =
                material.alphaMode == Material::AlphaMode::Blend;
            if ((render_transparents && isTransparent) ||
                (!render_transparents && !isTransparent))
            {
                const ModelInstance::PCBlock pcBlock{
                    .modelInstanceID = instance.id,
                    .meshID = subModel.meshID,
                    .materialID = subModel.materialID,
                };
                buffer.pushConstants(
                    _pipelineLayout,
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, // offset
                    sizeof(ModelInstance::PCBlock), &pcBlock);

                buffer.draw(mesh.indexCount(), 1, 0, 0);
            }
        }
    }

    buffer.endRendering();

    buffer.endDebugUtilsLabelEXT(); // Opaque

    buffer.end();

    return buffer;
}

bool Renderer::compileShaders(const World::DSLayouts &worldDSLayouts)
{
    fprintf(stderr, "Compiling Renderer shaders\n");

    std::string vertDefines;
    vertDefines += defineStr("CAMERA_SET", sCameraBindingSet);
    vertDefines += defineStr("VERTEX_BUFFERS_SET", sVertexBuffersBindingSet);
    vertDefines += defineStr("INDEX_BUFFERS_SET", sIndexBuffersBindingSet);
    vertDefines +=
        defineStr("MODEL_INSTANCE_TRFNS_SET", sModelInstanceTrfnsBindingSet);
    const auto vertSM =
        _device->compileShaderModule(Device::CompileShaderModuleArgs{
            .relPath = "shader/scene.vert",
            .debugName = "geometryVS",
            .defines = vertDefines,
        });

    std::string fragDefines;
    fragDefines += defineStr("LIGHTS_SET", sLightsBindingSet);
    fragDefines += defineStr("LIGHT_CLUSTERS_SET", sLightClustersBindingSet);
    fragDefines += defineStr("CAMERA_SET", sCameraBindingSet);
    fragDefines += defineStr("MATERIALS_SET", sMaterialsBindingSet);
    fragDefines +=
        defineStr("NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    const auto fragSM =
        _device->compileShaderModule(Device::CompileShaderModuleArgs{
            .relPath = "shader/scene.frag",
            .debugName = "geometryPS",
            .defines = fragDefines,
        });

    if (vertSM && fragSM)
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

    if (vertSM)
        _device->logical().destroy(*vertSM);
    if (fragSM)
        _device->logical().destroy(*fragSM);

    return false;
}

void Renderer::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        if (!_commandBuffers.empty())
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                asserted_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        destroyGraphicsPipelines();

        _device->destroy(_resources->images.sceneColor);
        _device->destroy(_resources->images.sceneDepth);

        _colorAttachments = {};
        _depthAttachments = {};
    }
}

void Renderer::destroyGraphicsPipelines()
{
    for (auto &p : _pipelines)
        _device->logical().destroy(p);
    _device->logical().destroy(_pipelineLayout);
}

void Renderer::createOutputs(const SwapchainConfig &swapConfig)
{
    {
        _resources->images.sceneColor = _device->createImage(ImageCreateInfo{
            .format = vk::Format::eR16G16B16A16Sfloat,
            .width = swapConfig.extent.width,
            .height = swapConfig.extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eColorAttachment | // Render
                          vk::ImageUsageFlagBits::eStorage,          // ToneMap
            .debugName = "sceneColor",
        });
    }
    {
        // Check depth buffer without stencil is supported
        const auto features =
            vk::FormatFeatureFlagBits::eDepthStencilAttachment;
        const auto properties =
            _device->physical().getFormatProperties(swapConfig.depthFormat);
        if ((properties.optimalTilingFeatures & features) != features)
            throw std::runtime_error("Depth format unsupported");

        _resources->images.sceneDepth = _device->createImage(ImageCreateInfo{
            .format = swapConfig.depthFormat,
            .width = swapConfig.extent.width,
            .height = swapConfig.extent.height,
            .usageFlags = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .debugName = "sceneDepth",
        });

        const auto commandBuffer = _device->beginGraphicsCommands();

        _resources->images.sceneDepth.transition(
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
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array<float, 4>{0.f, 0.f, 0.f, 0.f}},
    };
    _colorAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    _depthAttachments[0] = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array<float, 4>{1.f, 0.f, 0.f, 0.f}},
    };
    _depthAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

void Renderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // TODO: Dynamic viewport state?
    const vk::Viewport viewport{
        .x = 0.f,
        .y = 0.f,
        .width = static_cast<float>(swapConfig.extent.width),
        .height = static_cast<float>(swapConfig.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapConfig.extent,
    };
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

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

    std::array<vk::DescriptorSetLayout, 7> setLayouts = {};
    setLayouts[sLightsBindingSet] = worldDSLayouts.lights;
    setLayouts[sLightClustersBindingSet] =
        _resources->buffers.lightClusters.descriptorSetLayout;
    setLayouts[sCameraBindingSet] = camDSLayout;
    setLayouts[sMaterialsBindingSet] = worldDSLayouts.materialTextures;
    setLayouts[sVertexBuffersBindingSet] = worldDSLayouts.vertexBuffers;
    setLayouts[sIndexBuffersBindingSet] = worldDSLayouts.indexBuffers;
    setLayouts[sModelInstanceTrfnsBindingSet] = worldDSLayouts.modelInstances;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(ModelInstance::PCBlock),
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
                .layout = _pipelineLayout,
            },
            vk::PipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats =
                    &_resources->images.sceneColor.format,
                .depthAttachmentFormat = _resources->images.sceneDepth.format,
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

void Renderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            // Separate buffers for opaque and transparent
            .commandBufferCount = swapConfig.imageCount * 2,
        });
}
