#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

Renderer::Renderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating Renderer\n");

    if (!compileShaders())
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
    if (compileShaders())
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
    const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
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
        .pLabelName = "Opaque",
    });

    buffer.beginRendering(vk::RenderingInfo{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &_colorAttachment,
        .pDepthAttachment = &_depthAttachment,
    });

    // Draw opaque and alpha masked geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    const auto &scene = world._scenes[world._currentScene];

    const std::array<vk::DescriptorSet, 5> descriptorSets{
        scene.lights.descriptorSets[nextImage],
        cam.descriptorSet(nextImage),
        _resources->buffers.lightClusters.descriptorSets[nextImage],
        world._materialTexturesDS,
        scene.modelInstancesDescriptorSets[nextImage],
    };
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
            if (material.alphaMode != Material::AlphaMode::Blend)
            {
                const ModelInstance::PCBlock pcBlock{
                    .modelInstanceID = instance.id,
                    .materialID = subModel.materialID,
                };
                buffer.pushConstants(
                    _pipelineLayout,
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, // offset
                    sizeof(ModelInstance::PCBlock), &pcBlock);
                mesh.draw(buffer);
            }
        }
    }

    buffer.endRendering();

    buffer.endDebugUtilsLabelEXT(); // Opaque

    buffer.end();

    return buffer;
}

bool Renderer::compileShaders()
{
    fprintf(stderr, "Compiling Renderer shaders\n");

    const auto vertSM =
        _device->compileShaderModule("shader/scene.vert", "opaqueVS");
    const auto fragSM =
        _device->compileShaderModule("shader/scene.frag", "opaquePS");

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

        _colorAttachment = vk::RenderingAttachmentInfo{};
        _depthAttachment = vk::RenderingAttachmentInfo{};
    }
}

void Renderer::destroyGraphicsPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void Renderer::createOutputs(const SwapchainConfig &swapConfig)
{
    {
        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        _resources->images.sceneColor = _device->createImage(
            "sceneColor", vk::ImageType::e2D,
            vk::Extent3D{
                .width = swapConfig.extent.width,
                .height = swapConfig.extent.height,
                .depth = 1,
            },
            vk::Format::eR16G16B16A16Sfloat, subresourceRange,
            vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlagBits{},
            vk::ImageUsageFlagBits::eColorAttachment | // Render
                vk::ImageUsageFlagBits::eStorage,      // ToneMap
            vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    {
        // Check depth buffer without stencil is supported
        const auto features =
            vk::FormatFeatureFlagBits::eDepthStencilAttachment;
        const auto properties =
            _device->physical().getFormatProperties(swapConfig.depthFormat);
        if ((properties.optimalTilingFeatures & features) != features)
            throw std::runtime_error("Depth format unsupported");

        _resources->images.sceneDepth = _device->createImage(
            "sceneDepth", vk::ImageType::e2D,
            vk::Extent3D{
                .width = swapConfig.extent.width,
                .height = swapConfig.extent.height,
                .depth = 1,
            },
            swapConfig.depthFormat,
            vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlags{},
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

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
    _colorAttachment = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array<float, 4>{0.f, 0.f, 0.f, 0.f}},
    };
    _depthAttachment = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{std::array<float, 4>{1.f, 0.f, 0.f, 0.f}},
    };
}

void Renderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    const auto vertexBindingDescription = Vertex::bindingDescription();
    const auto vertexAttributeDescriptions = Vertex::attributeDescriptions();
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexBindingDescription,
        .vertexAttributeDescriptionCount =
            asserted_cast<uint32_t>(vertexAttributeDescriptions.size()),
        .pVertexAttributeDescriptions = vertexAttributeDescriptions.data(),
    };

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

    const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
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
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    const std::array<vk::DescriptorSetLayout, 5> setLayouts{
        worldDSLayouts.lights,
        camDSLayout,
        _resources->buffers.lightClusters.descriptorSetLayout,
        worldDSLayouts.materialTextures,
        worldDSLayouts.modelInstances,
    };
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
                .pColorBlendState = &colorBlendState,
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

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "Renderer",
            });
    }
}

void Renderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount,
        });
}
