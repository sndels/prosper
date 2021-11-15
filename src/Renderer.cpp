#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

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
    createRenderPass(swapConfig);
    createFramebuffer(swapConfig);
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::RenderPass Renderer::outputRenderpass() const { return _renderpass; }

vk::CommandBuffer Renderer::execute(
    const World &world, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage) const
{
    updateUniformBuffers(world, cam, nextImage);

    return recordCommandBuffer(world, cam, renderArea, nextImage);
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

        _device->logical().destroy(_pipelines.pbr);
        _device->logical().destroy(_pipelines.pbrAlphaBlend);
        _device->logical().destroy(_pipelines.skybox);
        _device->logical().destroy(_pipelineLayouts.pbr);
        _device->logical().destroy(_pipelineLayouts.skybox);
        _device->logical().destroy(_fbo);
        _device->destroy(_resources->images.sceneColor);
        _device->destroy(_resources->images.sceneDepth);
        _device->logical().destroy(_renderpass);
    }
}

void Renderer::createRenderPass(const SwapchainConfig &swapConfig)
{
    const std::array<vk::AttachmentDescription, 2> attachments = {
        // color
        vk::AttachmentDescription{
            .format = swapConfig.surfaceFormat.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal},
        vk::AttachmentDescription{
            // depth
            .format = swapConfig.depthFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
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

    // Synchronize and handle layout transitions
    const std::array<vk::SubpassDependency, 2> dependencies{
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite},
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eTopOfPipe,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead}};

    _renderpass = _device->logical().createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data()});
}

void Renderer::createFramebuffer(const SwapchainConfig &swapConfig)
{
    {
        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};

        _resources->images.sceneColor = _device->createImage(
            "sceneColor", swapConfig.extent, swapConfig.surfaceFormat.format,
            subresourceRange, vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlagBits{},
            vk::ImageUsageFlagBits::eColorAttachment |
                vk::ImageUsageFlagBits::eTransferSrc,
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

        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};

        _resources->images.sceneDepth = _device->createImage(
            "sceneDepth", swapConfig.extent, swapConfig.depthFormat,
            subresourceRange, vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlags{},
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY);

        const auto commandBuffer = _device->beginGraphicsCommands();

        transitionImageLayout(
            commandBuffer, _resources->images.sceneDepth.handle,
            subresourceRange, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::AccessFlags{},
            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eEarlyFragmentTests);

        _device->endGraphicsCommands(commandBuffer);
    }
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
    {
        const auto vertSPV = readFileBytes(binPath("shader/shader.vert.spv"));
        const auto fragSPV = readFileBytes(binPath("shader/shader.frag.spv"));
        const vk::ShaderModule vertSM =
            createShaderModule(_device->logical(), vertSPV);
        const vk::ShaderModule fragSM =
            createShaderModule(_device->logical(), fragSPV);
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
        const auto vertexAttributeDescriptions =
            Vertex::attributeDescriptions();
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

        // Alpha blend is created with a modified version
        vk::PipelineRasterizationStateCreateInfo rasterizerState{
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

        // Alpha blend pipeline is created with a modified version
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA};
        const vk::PipelineColorBlendStateCreateInfo colorBlendState{
            .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

        const std::array<vk::DescriptorSetLayout, 3> setLayouts = {
            {camDSLayout, worldDSLayouts.modelInstance,
             worldDSLayouts.material}};
        const vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(Material::PCBlock)};
        _pipelineLayouts.pbr = _device->logical().createPipelineLayout(
            vk::PipelineLayoutCreateInfo{
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
            .layout = _pipelineLayouts.pbr,
            .renderPass = _renderpass,
            .subpass = 0};

        {
            auto pipeline = _device->logical().createGraphicsPipeline(
                vk::PipelineCache{}, createInfo);
            if (pipeline.result != vk::Result::eSuccess)
                throw std::runtime_error("Failed to create pbr pipeline");

            _pipelines.pbr = pipeline.value;
        }

        rasterizerState.cullMode = vk::CullModeFlagBits::eNone;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        colorBlendAttachment.dstColorBlendFactor =
            vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
        colorBlendAttachment.srcAlphaBlendFactor =
            vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;

        {
            auto pipeline = _device->logical().createGraphicsPipeline(
                vk::PipelineCache{}, createInfo);
            if (pipeline.result != vk::Result::eSuccess)
                throw std::runtime_error(
                    "Failed to create pbr alpha blend pipeline");

            _pipelines.pbrAlphaBlend = pipeline.value;
        }

        _device->logical().destroyShaderModule(vertSM);
        _device->logical().destroyShaderModule(fragSM);
    }

    {
        const auto vertSPV = readFileBytes(binPath("shader/skybox.vert.spv"));
        const auto fragSPV = readFileBytes(binPath("shader/skybox.frag.spv"));
        const vk::ShaderModule vertSM =
            createShaderModule(_device->logical(), vertSPV);
        const vk::ShaderModule fragSM =
            createShaderModule(_device->logical(), fragSPV);
        const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertSM,
                .pName = "main"},
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragSM,
                .pName = "main"}};

        const vk::VertexInputBindingDescription vertexBindingDescription{
            .binding = 0,
            .stride = sizeof(vec3), // Only position
            .inputRate = vk::VertexInputRate::eVertex};
        const vk::VertexInputAttributeDescription vertexAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = 0};
        const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBindingDescription,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &vertexAttributeDescription};

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
            .maxDepth = 1.f};
        const vk::Rect2D scissor{.offset = {0, 0}, .extent = swapConfig.extent};
        const vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor};

        const vk::PipelineRasterizationStateCreateInfo rasterizerState{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode =
                vk::CullModeFlagBits::eNone, // Draw the skybox from inside
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0};

        const vk::PipelineMultisampleStateCreateInfo multisampleState{
            .rasterizationSamples = vk::SampleCountFlagBits::e1};

        const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLessOrEqual};

        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA};
        const vk::PipelineColorBlendStateCreateInfo colorBlendState{
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment};

        _pipelineLayouts.skybox = _device->logical().createPipelineLayout(
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1, .pSetLayouts = &worldDSLayouts.skybox});

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
            .layout = _pipelineLayouts.skybox,
            .renderPass = _renderpass,
            .subpass = 0};
        {
            auto pipeline = _device->logical().createGraphicsPipeline(
                vk::PipelineCache{}, createInfo);
            if (pipeline.result != vk::Result::eSuccess)
                throw std::runtime_error("Failed to create skybox pipeline");
            _pipelines.skybox = pipeline.value;
        }

        _device->logical().destroyShaderModule(vertSM);
        _device->logical().destroyShaderModule(fragSM);
    }
}

void Renderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}

void Renderer::updateUniformBuffers(
    const World &world, const Camera &cam, const uint32_t nextImage) const
{
    cam.updateBuffer(nextImage);

    const mat4 worldToClip =
        cam.cameraToClip() * mat4(mat3(cam.worldToCamera()));
    void *data;
    _device->map(world._skyboxUniformBuffers[nextImage].allocation, &data);
    memcpy(data, &worldToClip, sizeof(mat4));
    _device->unmap(world._skyboxUniformBuffers[nextImage].allocation);

    for (const auto &instance : world.currentScene().modelInstances)
        instance.updateBuffer(_device, nextImage);
}

vk::CommandBuffer Renderer::recordCommandBuffer(
    const World &world, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

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

    // Draw opaque and alpha masked geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.pbr);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
        0, // firstSet
        1, &cam.descriptorSet(nextImage), 0, nullptr);

    recordModelInstances(
        buffer, nextImage, world.currentScene().modelInstances,
        [](const Mesh &mesh)
        { return mesh.material()._alphaMode == Material::AlphaMode::Blend; });

    // Skybox doesn't need to be drawn under opaque geometry but should be
    // before transparents
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.skybox);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayouts.skybox,
        0, // firstSet
        1, &world._skyboxDSs[nextImage], 0, nullptr);

    world.drawSkybox(buffer);

    // Draw transparent geometry
    buffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics, _pipelines.pbrAlphaBlend);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
        0, // firstSet
        1, &cam.descriptorSet(nextImage), 0, nullptr);

    // TODO: Sort back to front
    recordModelInstances(
        buffer, nextImage, world.currentScene().modelInstances,
        [](const Mesh &mesh)
        { return mesh.material()._alphaMode != Material::AlphaMode::Blend; });

    buffer.endRenderPass();

    buffer.end();

    return buffer;
}

void Renderer::recordModelInstances(
    const vk::CommandBuffer buffer, const uint32_t nextImage,
    const std::vector<Scene::ModelInstance> &instances,
    const std::function<bool(const Mesh &)> &cullMesh) const
{
    for (const auto &instance : instances)
    {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
            1, // firstSet
            1, &instance.descriptorSets[nextImage], 0, nullptr);
        for (const auto &mesh : instance.model->_meshes)
        {
            if (cullMesh(mesh))
                continue;
            buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
                2, // firstSet
                1, &mesh.material()._descriptorSet, 0, nullptr);
            const auto pcBlock = mesh.material().pcBlock();
            buffer.pushConstants(
                _pipelineLayouts.pbr, vk::ShaderStageFlagBits::eFragment,
                0, // offset
                sizeof(Material::PCBlock), &pcBlock);
            mesh.draw(buffer);
        }
    }
}
