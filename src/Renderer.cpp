#include "Renderer.hpp"

#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

#include "Constants.hpp"
#include "VkUtils.hpp"

using namespace glm;

namespace {
static std::vector<char> readFile(const std::string &filename) {
    // Open from end to find size from initial position
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(std::string{"Failed to open file '"} +
                                 filename + "'");

    const auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    // Seek to beginning and read
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

vk::ShaderModule createShaderModule(const vk::Device device,
                                    const std::vector<char> &spv) {
    return device.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spv.size(),
        .pCode = reinterpret_cast<const uint32_t *>(spv.data())});
}
} // namespace

Renderer::Renderer(std::shared_ptr<Device> device,
                   const SwapchainConfig &swapConfig,
                   const vk::DescriptorSetLayout camDSLayout,
                   const World::DSLayouts &worldDSLayouts)
    : _device{device} {
    // Semaphores correspond to logical frames instead of swapchain images
    createSemaphores(MAX_FRAMES_IN_FLIGHT);
    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

Renderer::~Renderer() {
    if (_device) {
        for (auto &semaphore : _renderFinishedSemaphores)
            _device->logical().destroy(semaphore);
        for (auto &semaphore : _imageAvailableSemaphores)
            _device->logical().destroy(semaphore);
        destroySwapchainRelated();
    }
}

void Renderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts) {
    destroySwapchainRelated();
    createRenderPass(swapConfig);
    createFramebuffer(swapConfig);
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::Semaphore Renderer::imageAvailable(const uint32_t frame) const {
    return _imageAvailableSemaphores[frame];
}

vk::RenderPass Renderer::outputRenderpass() const { return _renderpass; }

std::array<vk::Semaphore, 1>
Renderer::drawFrame(const World &world, const Camera &cam,
                    const Swapchain &swapchain,
                    const uint32_t nextImage) const {
    updateUniformBuffers(world, cam, nextImage);

    recordCommandBuffer(world, cam, swapchain, nextImage);

    // Submit queue
    const size_t nextFrame = swapchain.nextFrame();

    const std::array<vk::Semaphore, 1> waitSemaphores = {
        _imageAvailableSemaphores[nextFrame]};
    const std::array<vk::PipelineStageFlags, 1> waitStages = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const std::array<vk::Semaphore, 1> signalSemaphores = {
        _renderFinishedSemaphores[nextFrame]};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = waitSemaphores.size(),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = 1, // commandBufferCount
        .pCommandBuffers = &_commandBuffers[nextImage],
        .signalSemaphoreCount = signalSemaphores.size(),
        .pSignalSemaphores = signalSemaphores.data()};
    _device->graphicsQueue().submit(1, &submitInfo, swapchain.currentFence());

    return signalSemaphores;
}

void Renderer::destroySwapchainRelated() {
    if (_device) {
        _device->logical().freeCommandBuffers(_device->graphicsPool(),
                                              _commandBuffers.size(),
                                              _commandBuffers.data());

        _device->logical().destroy(_pipelines.pbr);
        _device->logical().destroy(_pipelines.pbrAlphaBlend);
        _device->logical().destroy(_pipelines.skybox);
        _device->logical().destroy(_pipelineLayouts.pbr);
        _device->logical().destroy(_pipelineLayouts.skybox);
        _device->logical().destroy(_fbo);
        _device->destroy(_depthImage);
        _device->destroy(_colorImage);
        _device->logical().destroy(_renderpass);
    }
}

void Renderer::createRenderPass(const SwapchainConfig &swapConfig) {
    // TODO: Can swap surface formats change after first creation?
    const std::array<vk::AttachmentDescription, 2> attachments = {
        // swap color
        vk::AttachmentDescription{
            .format = swapConfig.surfaceFormat.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout =
                vk::ImageLayout::eTransferSrcOptimal // will be blitted from
        },
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
        .attachmentCount = attachments.size(),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data()});
}

void Renderer::createFramebuffer(const SwapchainConfig &swapConfig) {
    {
        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};

        _colorImage = _device->createImage(
            swapConfig.extent, swapConfig.surfaceFormat.format,
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

        _depthImage = _device->createImage(
            swapConfig.extent, swapConfig.depthFormat, subresourceRange,
            vk::ImageViewType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageCreateFlags{},
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            VMA_MEMORY_USAGE_GPU_ONLY);

        const auto commandBuffer = _device->beginGraphicsCommands();

        transitionImageLayout(commandBuffer, _depthImage.handle,
                              subresourceRange, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal,
                              vk::AccessFlags{},
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                              vk::PipelineStageFlagBits::eTopOfPipe,
                              vk::PipelineStageFlagBits::eEarlyFragmentTests);

        _device->endGraphicsCommands(commandBuffer);
    }
    const std::array<vk::ImageView, 2> attachments = {
        {_colorImage.view, _depthImage.view}};
    _fbo = _device->logical().createFramebuffer(
        vk::FramebufferCreateInfo{.renderPass = _renderpass,
                                  .attachmentCount = attachments.size(),
                                  .pAttachments = attachments.data(),
                                  .width = swapConfig.extent.width,
                                  .height = swapConfig.extent.height,
                                  .layers = 1});

    // Fbo layers and extent match swap image for now
    const vk::ImageSubresourceLayers layers{.aspectMask =
                                                vk::ImageAspectFlagBits::eColor,
                                            .mipLevel = 0,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1};
    const std::array<vk::Offset3D, 2> offsets{
        {{0},
         {static_cast<int32_t>(swapConfig.extent.width),
          static_cast<int32_t>(swapConfig.extent.height), 1}}};
    _fboToSwap = vk::ImageBlit{
        .srcSubresource = layers,
        .srcOffsets = offsets,
        .dstSubresource = layers,
        .dstOffsets = offsets,
    };
}

void Renderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts) {
    {
        const auto vertSPV = readFile(binPath("shader/shader.vert.spv"));
        const auto fragSPV = readFile(binPath("shader/shader.frag.spv"));
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
                vertexAttributeDescriptions.size(),
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
            vk::PipelineLayoutCreateInfo{.setLayoutCount = setLayouts.size(),
                                         .pSetLayouts = setLayouts.data(),
                                         .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &pcRange});

        const vk::GraphicsPipelineCreateInfo createInfo{
            .stageCount = shaderStages.size(),
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
        const auto vertSPV = readFile(binPath("shader/skybox.vert.spv"));
        const auto fragSPV = readFile(binPath("shader/skybox.frag.spv"));
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
            .stageCount = shaderStages.size(),
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

void Renderer::createCommandBuffers(const SwapchainConfig &swapConfig) {
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}

void Renderer::createSemaphores(const uint32_t concurrentFrameCount) {
    for (size_t i = 0; i < concurrentFrameCount; ++i) {
        _imageAvailableSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
        _renderFinishedSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
    }
}

void Renderer::updateUniformBuffers(const World &world, const Camera &cam,
                                    const uint32_t nextImage) const {
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

void Renderer::recordCommandBuffer(const World &world, const Camera &cam,
                                   const Swapchain &swapchain,
                                   const uint32_t nextImage) const {
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});

    const std::array<vk::ClearValue, 2> clearColors = {
        {vk::ClearValue{std::array<float, 4>{0.f, 0.f, 0.f, 0.f}}, // color
         vk::ClearValue{
             std::array<float, 4>{1.f, 0.f, 0.f, 0.f}}} // depth stencil
    };
    buffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = _renderpass,
            .framebuffer = _fbo,
            .renderArea =
                vk::Rect2D{.offset = {0, 0}, .extent = swapchain.extent()},
            .clearValueCount = clearColors.size(),
            .pClearValues = clearColors.data()},
        vk::SubpassContents::eInline);

    // Draw opaque and alpha masked geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.pbr);

    buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                              _pipelineLayouts.pbr,
                              0, // firstSet
                              1, &cam.descriptorSet(nextImage), 0, nullptr);

    recordModelInstances(buffer, nextImage, world.currentScene().modelInstances,
                         [](const Mesh &mesh) {
                             return mesh.material()._alphaMode ==
                                    Material::AlphaMode::Blend;
                         });

    // Skybox doesn't need to be drawn under opaque geometry but should be
    // before transparents
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.skybox);

    buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                              _pipelineLayouts.skybox,
                              0, // firstSet
                              1, &world._skyboxDSs[nextImage], 0, nullptr);

    world.drawSkybox(buffer);

    // Draw transparent geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                        _pipelines.pbrAlphaBlend);

    buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                              _pipelineLayouts.pbr,
                              0, // firstSet
                              1, &cam.descriptorSet(nextImage), 0, nullptr);

    // TODO: Sort back to front
    recordModelInstances(buffer, nextImage, world.currentScene().modelInstances,
                         [](const Mesh &mesh) {
                             return mesh.material()._alphaMode !=
                                    Material::AlphaMode::Blend;
                         });

    buffer.endRenderPass();

    // Blit to support different internal rendering resolution (and color
    // format?) the future
    const auto &swapImage = swapchain.image(nextImage);

    transitionImageLayout(buffer, swapImage.handle, swapImage.subresourceRange,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
                          vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer);

    buffer.blitImage(_colorImage.handle, vk::ImageLayout::eTransferSrcOptimal,
                     swapImage.handle, vk::ImageLayout::eTransferDstOptimal, 1,
                     &_fboToSwap, vk::Filter::eLinear);

    transitionImageLayout(
        buffer, swapImage.handle, swapImage.subresourceRange,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer);

    buffer.end();
}

void Renderer::recordModelInstances(
    const vk::CommandBuffer buffer, const uint32_t nextImage,
    const std::vector<Scene::ModelInstance> &instances,
    const std::function<bool(const Mesh &)> &cullMesh) const {
    for (const auto &instance : instances) {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
            1, // firstSet
            1, &instance.descriptorSets[nextImage], 0, nullptr);
        for (const auto &mesh : instance.model->_meshes) {
            if (cullMesh(mesh))
                continue;
            buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, _pipelineLayouts.pbr,
                2, // firstSet
                1, &mesh.material()._descriptorSet, 0, nullptr);
            const auto pcBlock = mesh.material().pcBlock();
            buffer.pushConstants(_pipelineLayouts.pbr,
                                 vk::ShaderStageFlagBits::eFragment,
                                 0, // offset
                                 sizeof(Material::PCBlock), &pcBlock);
            mesh.draw(buffer);
        }
    }
}
