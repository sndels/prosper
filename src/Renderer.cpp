#include "Renderer.hpp"

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

Renderer::Renderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

Renderer::~Renderer()
{
    if (_device)
    {
        destroySwapchainRelated();
    }
}

void Renderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createOutputs(swapConfig);
    createRenderPass();
    createFramebuffer(swapConfig);
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer Renderer::recordCommandBuffer(
    const Scene &scene, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const std::array<vk::ImageMemoryBarrier2KHR, 2> barriers{
        _resources->images.sceneColor.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput,
            .accessMask = vk::AccessFlagBits2KHR::eColorAttachmentWrite,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        }),
        _resources->images.sceneDepth.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests,
            .accessMask = vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite,
            .layout = vk::ImageLayout::eDepthAttachmentOptimal,
        }),
    };

    buffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });

    const std::array<vk::ClearValue, 2> clearColors = {
        {vk::ClearValue{std::array<float, 4>{0.f, 0.f, 0.f, 0.f}}, // color
         vk::ClearValue{
             std::array<float, 4>{1.f, 0.f, 0.f, 0.f}}} // depth stencil
    };
    buffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = _renderpass,
            .framebuffer = _fbo,
            .renderArea = renderArea,
            .clearValueCount = static_cast<uint32_t>(clearColors.size()),
            .pClearValues = clearColors.data()},
        vk::SubpassContents::eInline);

    buffer.beginDebugUtilsLabelEXT(
        vk::DebugUtilsLabelEXT{.pLabelName = "Opaque"});

    // Draw opaque and alpha masked geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    const std::array<vk::DescriptorSet, 2> descriptorSets{
        scene.lights.descriptorSets[nextImage], cam.descriptorSet(nextImage)};
    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0,
        nullptr);

    recordModelInstances(
        buffer, nextImage, scene.modelInstances,
        [](const Mesh &mesh)
        { return mesh.material()._alphaMode != Material::AlphaMode::Blend; });

    buffer.endDebugUtilsLabelEXT(); // Opaque

    buffer.endRenderPass();

    buffer.end();

    return buffer;
}

void Renderer::destroySwapchainRelated()
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

        _device->logical().destroy(_pipeline);
        _device->logical().destroy(_pipelineLayout);
        _device->logical().destroy(_fbo);
        _device->logical().destroy(_renderpass);
        _device->destroy(_resources->images.sceneColor);
        _device->destroy(_resources->images.sceneDepth);
    }
}

void Renderer::createOutputs(const SwapchainConfig &swapConfig)
{
    {
        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};

        _resources->images.sceneColor = _device->createImage(
            "sceneColor", swapConfig.extent, vk::Format::eR16G16B16A16Sfloat,
            subresourceRange, vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlagBits{},
            vk::ImageUsageFlagBits::eColorAttachment | // Render
                vk::ImageUsageFlagBits::eStorage,      // ToneMap
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY);
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
            "sceneDepth", swapConfig.extent, swapConfig.depthFormat,
            vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1},
            vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlags{},
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY);

        const auto commandBuffer = _device->beginGraphicsCommands();

        _resources->images.sceneDepth.transition(
            commandBuffer,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests,
                .accessMask =
                    vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            });

        _device->endGraphicsCommands(commandBuffer);
    }
}

void Renderer::createRenderPass()
{
    const std::array<vk::AttachmentDescription, 2> attachments = {
        // color
        vk::AttachmentDescription{
            .format = _resources->images.sceneColor.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal},
        vk::AttachmentDescription{
            // depth
            .format = _resources->images.sceneDepth.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal}};
    const vk::AttachmentReference swapAttachmentRef{
        .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference depthAttachmentRef{
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    // Output
    const vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &swapAttachmentRef,
        .pDepthStencilAttachment = &depthAttachmentRef};

    _renderpass = _device->logical().createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass});

    _device->logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eRenderPass,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkRenderPass>(_renderpass)),
            .pObjectName = "Renderer"});
}

void Renderer::createFramebuffer(const SwapchainConfig &swapConfig)
{
    const std::array<vk::ImageView, 2> attachments = {
        {_resources->images.sceneColor.view,
         _resources->images.sceneDepth.view}};
    _fbo = _device->logical().createFramebuffer(vk::FramebufferCreateInfo{
        .renderPass = _renderpass,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = swapConfig.extent.width,
        .height = swapConfig.extent.height,
        .layers = 1});
}

void Renderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    const auto vertSPV = readFileBytes(binPath("shader/scene.vert.spv"));
    const auto fragSPV = readFileBytes(binPath("shader/scene.frag.spv"));
    const vk::ShaderModule vertSM =
        createShaderModule(_device->logical(), "opaqueVS", vertSPV);
    const vk::ShaderModule fragSM =
        createShaderModule(_device->logical(), "opaquePS", fragSPV);
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertSM,
            .pName = "main"},
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragSM,
            .pName = "main"}};

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
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0};

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1};

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess};

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
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

    const std::array<vk::DescriptorSetLayout, 4> setLayouts{
        worldDSLayouts.lights,
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

    const vk::GraphicsPipelineCreateInfo createInfo{
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizerState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &colorBlendState,
        .layout = _pipelineLayout,
        .renderPass = _renderpass,
        .subpass = 0};

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;
    }

    _device->logical().destroyShaderModule(vertSM);
    _device->logical().destroyShaderModule(fragSM);
}

void Renderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}

void Renderer::recordModelInstances(
    const vk::CommandBuffer buffer, const uint32_t nextImage,
    const std::vector<Scene::ModelInstance> &instances,
    const std::function<bool(const Mesh &)> &shouldRender) const
{
    for (const auto &instance : instances)
    {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            2, // firstSet
            1, &instance.descriptorSets[nextImage], 0, nullptr);
        for (const auto &mesh : instance.model->_meshes)
        {
            if (shouldRender(mesh))
            {
                buffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics, _pipelineLayout,
                    3, // firstSet
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
