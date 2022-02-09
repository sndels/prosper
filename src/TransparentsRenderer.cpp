#include "TransparentsRenderer.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/gtc/matrix_transform.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

TransparentsRenderer::TransparentsRenderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating TransparentsRenderer\n");

    if (!compileShaders())
        throw std::runtime_error(
            "TransparentsRenderer shader compilation failed");
    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

TransparentsRenderer::~TransparentsRenderer()
{
    if (_device)
    {
        destroySwapchainRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void TransparentsRenderer::recompileShaders(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders())
    {
        destroyGraphicsPipeline();
        createGraphicsPipeline(swapConfig, camDSLayout, worldDSLayouts);
    }
}

void TransparentsRenderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createAttachments();
    createGraphicsPipeline(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer TransparentsRenderer::recordCommandBuffer(
    const Scene &scene, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const std::array<vk::ImageMemoryBarrier2KHR, 3> imageBarriers{
        _resources->images.sceneColor.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput,
            .accessMask = vk::AccessFlagBits2KHR::eColorAttachmentRead,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        }),
        _resources->images.sceneDepth.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests,
            .accessMask = vk::AccessFlagBits2KHR::eDepthStencilAttachmentRead,
            .layout = vk::ImageLayout::eDepthAttachmentOptimal,
        }),
        _resources->buffers.lightClusters.pointers.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eFragmentShader,
            .accessMask = vk::AccessFlagBits2KHR::eShaderRead,
            .layout = vk::ImageLayout::eGeneral,
        }),
    };

    const std::array<vk::BufferMemoryBarrier2KHR, 2> bufferBarriers{
        _resources->buffers.lightClusters.indicesCount.transitionBarrier(
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2KHR::eComputeShader,
                .accessMask = vk::AccessFlagBits2KHR::eShaderRead,
            }),
        _resources->buffers.lightClusters.indices.transitionBarrier(BufferState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eComputeShader,
            .accessMask = vk::AccessFlagBits2KHR::eShaderRead,
        }),
    };

    buffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .bufferMemoryBarrierCount =
            static_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    });

    buffer.beginDebugUtilsLabelEXT(
        vk::DebugUtilsLabelEXT{.pLabelName = "Transparents"});

    buffer.beginRenderingKHR(vk::RenderingInfoKHR{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &_colorAttachment,
        .pDepthAttachment = &_depthAttachment,
    });

    // Draw transparent geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    const std::array<vk::DescriptorSet, 3> descriptorSets{
        scene.lights.descriptorSets[nextImage],
        _resources->buffers.lightClusters.descriptorSets[nextImage],
        cam.descriptorSet(nextImage)};
    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0,
        nullptr);

    // TODO: Sort back to front
    recordModelInstances(
        buffer, nextImage, scene.modelInstances,
        [](const Mesh &mesh)
        { return mesh.material()._alphaMode == Material::AlphaMode::Blend; });

    buffer.endRenderingKHR();

    buffer.endDebugUtilsLabelEXT(); // Transparents

    buffer.end();

    return buffer;
}

bool TransparentsRenderer::compileShaders()
{
    fprintf(stderr, "Compiling TransparentsRenderer shaders\n");

    const auto vertSM =
        _device->compileShaderModule("shader/scene.vert", "transparentsVS");
    const auto fragSM =
        _device->compileShaderModule("shader/scene.frag", "transparentsPS");

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

void TransparentsRenderer::destroySwapchainRelated()
{
    if (_device)
    {
        if (_commandBuffers.size() > 0)
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                static_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        destroyGraphicsPipeline();

        _colorAttachment = vk::RenderingAttachmentInfoKHR{};
        _depthAttachment = vk::RenderingAttachmentInfoKHR{};
    }
}

void TransparentsRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void TransparentsRenderer::createAttachments()
{
    _colorAttachment = vk::RenderingAttachmentInfoKHR{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    _depthAttachment = vk::RenderingAttachmentInfoKHR{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

void TransparentsRenderer::createGraphicsPipeline(
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
            static_cast<uint32_t>(vertexAttributeDescriptions.size()),
        .pVertexAttributeDescriptions = vertexAttributeDescriptions.data()};

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};

    // TODO: Dynamic viewport state?
    const vk::Viewport viewport{
        .x = 0.f,
        .y = 0.f,
        .width = static_cast<float>(swapConfig.extent.width),
        .height = static_cast<float>(swapConfig.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f};
    const vk::Rect2D scissor{.offset = {0, 0}, .extent = swapConfig.extent};
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    const vk::PipelineRasterizationStateCreateInfo rasterizerState{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0};

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1};

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess};

    const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

    const std::array<vk::DescriptorSetLayout, 5> setLayouts{
        worldDSLayouts.lights,
        _resources->buffers.lightClusters.descriptorSetLayout,
        camDSLayout,
        worldDSLayouts.modelInstance,
        worldDSLayouts.material,
    };
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(Material::PCBlock)};
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange});

    const vk::PipelineRenderingCreateInfoKHR renderingCreateInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &_resources->images.sceneColor.format,
        .depthAttachmentFormat = _resources->images.sceneDepth.format,
    };

    const vk::GraphicsPipelineCreateInfo createInfo{
        .pNext = &renderingCreateInfo,
        .stageCount = static_cast<uint32_t>(_shaderStages.size()),
        .pStages = _shaderStages.data(),
        .pVertexInputState = &vertInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizerState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &colorBlendState,
        .layout = _pipelineLayout,
    };

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error(
                "Failed to create pbr alpha blend pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "TransparentsRendering",
            });
    }
}

void TransparentsRenderer::createCommandBuffers(
    const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}

void TransparentsRenderer::recordModelInstances(
    const vk::CommandBuffer buffer, const uint32_t nextImage,
    const std::vector<Scene::ModelInstance> &instances,
    const std::function<bool(const Mesh &)> &shouldRender) const
{
    for (const auto &instance : instances)
    {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            3, // firstSet
            1, &instance.descriptorSets[nextImage], 0, nullptr);
        for (const auto &mesh : instance.model->_meshes)
        {
            if (shouldRender(mesh))
            {
                buffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics, _pipelineLayout,
                    4, // firstSet
                    1, &mesh.material()._descriptorSet, 0, nullptr);
                const auto pcBlock = mesh.material().pcBlock();
                buffer.pushConstants(
                    _pipelineLayout, vk::ShaderStageFlagBits::eFragment,
                    0, // offset
                    sizeof(Material::PCBlock), &pcBlock);
                mesh.draw(buffer);
            }
        }
    }
}
