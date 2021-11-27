#include "SkyboxRenderer.hpp"

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

SkyboxRenderer::SkyboxRenderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    recreateSwapchainRelated(swapConfig, worldDSLayouts);
}

SkyboxRenderer::~SkyboxRenderer()
{
    if (_device)
    {
        destroySwapchainRelated();
    }
}

void SkyboxRenderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createRenderPass();
    createFramebuffer(swapConfig);
    createGraphicsPipelines(swapConfig, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer SkyboxRenderer::recordCommandBuffer(
    const World &world, const vk::Rect2D &renderArea,
    const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    _resources->images.sceneColor.transitionBarrier(
        buffer, vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    _resources->images.sceneDepth.transitionBarrier(
        buffer, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits::eDepthStencilAttachmentRead,
        vk::PipelineStageFlagBits::eEarlyFragmentTests);

    buffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = _renderpass,
            .framebuffer = _fbo,
            .renderArea = renderArea,
        },
        vk::SubpassContents::eInline);

    buffer.beginDebugUtilsLabelEXT(
        vk::DebugUtilsLabelEXT{.pLabelName = "Skybox"});

    // Skybox doesn't need to be drawn under opaque geometry but should be
    // before transparents
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        1, &world._skyboxDSs[nextImage], 0, nullptr);

    world.drawSkybox(buffer);

    buffer.endDebugUtilsLabelEXT(); // Skybox

    buffer.endRenderPass();

    buffer.end();

    return buffer;
}

void SkyboxRenderer::destroySwapchainRelated()
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
    }
}

void SkyboxRenderer::createRenderPass()
{
    const std::array<vk::AttachmentDescription, 2> attachments = {
        // color
        vk::AttachmentDescription{
            .format = _resources->images.sceneColor.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal},
        vk::AttachmentDescription{
            // depth
            .format = _resources->images.sceneDepth.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
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
            .pObjectName = "SkyboxRenderer"});
}

void SkyboxRenderer::createFramebuffer(const SwapchainConfig &swapConfig)
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

void SkyboxRenderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
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
        .cullMode = vk::CullModeFlagBits::eNone, // Draw the skybox from inside
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
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment};

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
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
        .layout = _pipelineLayout,
        .renderPass = _renderpass,
        .subpass = 0};
    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create skybox pipeline");
        _pipeline = pipeline.value;
    }

    _device->logical().destroyShaderModule(vertSM);
    _device->logical().destroyShaderModule(fragSM);
}

void SkyboxRenderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}
