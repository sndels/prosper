#include "App.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "Constants.hpp"

namespace {
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    static std::vector<char> readFile(const std::string& filename)
    {
        // Open from end to find size from initial position
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw  std::runtime_error(std::string("Failed to open file '") + filename + "'");

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        // Seek to beginning and read
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }

    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& spv)
    {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spv.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(spv.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule))
            throw std::runtime_error("Failed to create shader module");

        return shaderModule;
    }
}

App::~App()
{
    // Destroy vulkan stuff
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(_device.handle(), _renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(_device.handle(), _imageAvailableSemaphores[i], nullptr);
    }
    destroySwapchainAndRelated();
}

void App::init()
{
    _window.init(WIDTH, HEIGHT, "prosper");

    // Init vulkan
    _device.init(_window.ptr());

    createSwapchainAndRelated();

    createSemaphores();
}

void App::run() 
{
    while (_window.open()) {
        _window.startFrame();
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    vkDeviceWaitIdle(_device.handle());
}

void App::destroySwapchainAndRelated()
{
    // Destroy vulkan resources
    vkFreeCommandBuffers(_device.handle(), _device.commandPool(), static_cast<uint32_t>(_vkCommandBuffers.size()), _vkCommandBuffers.data());
    vkDestroyPipeline(_device.handle(), _vkGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_device.handle(), _vkGraphicsPipelineLayout, nullptr);
    vkDestroyRenderPass(_device.handle(), _vkRenderPass, nullptr);

    // Also clear the handles
    _vkRenderPass = VK_NULL_HANDLE;
    _vkGraphicsPipelineLayout = VK_NULL_HANDLE;
    _vkGraphicsPipeline = VK_NULL_HANDLE;
    _vkCommandBuffers.clear();

    // Don't forget the actual swapchain
    _swapchain.destroy();
}

void App::createSwapchainAndRelated()
{
    SwapchainConfig swapConfig = selectSwapchainConfig(&_device, {_window.width(), _window.height()});

    createRenderPass(swapConfig);
    createGraphicsPipeline(swapConfig);

    _swapchain.create(&_device, _vkRenderPass, swapConfig);

    createCommandBuffers();
}

void App::recreateSwapchainAndRelated()
{
    while (_window.width() == 0 && _window.height() == 0) {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    vkDeviceWaitIdle(_device.handle());

    destroySwapchainAndRelated();
    createSwapchainAndRelated();
}

void App::createRenderPass(const SwapchainConfig& swapConfig)
{
    // Fill color attachment data for the swap buffer
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapConfig.surfaceFormat.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Fill reference to it
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Create subpass for output
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Create subpass dependency from last pass to synchronize render passess
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask =  0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Create render pass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(_device.handle(), &renderPassInfo, nullptr, &_vkRenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");
}

void App::createGraphicsPipeline(const SwapchainConfig& swapConfig)
{
    // Create modules for shaders
    auto vertSPV = readFile("shader/shader.vert.spv");
    auto fragSPV = readFile("shader/shader.frag.spv");
    VkShaderModule vertShaderModule = createShaderModule(_device.handle(), vertSPV);
    VkShaderModule fragShaderModule = createShaderModule(_device.handle(), fragSPV);

    // Fill out create infos for the shader stages
    VkPipelineShaderStageCreateInfo vertStageInfo = {};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName = "main";
    vertStageInfo.pSpecializationInfo = nullptr; // optional, can set shader constants

    VkPipelineShaderStageCreateInfo fragStageInfo = {};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    // Fill out shader stage inputs
    VkPipelineVertexInputStateCreateInfo vertInputInfo = {};
    vertInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertInputInfo.vertexBindingDescriptionCount = 0;
    vertInputInfo.pVertexBindingDescriptions = nullptr;
    vertInputInfo.vertexAttributeDescriptionCount = 0;
    vertInputInfo.pVertexAttributeDescriptions = nullptr;

    // Fill out input topology
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Set up viewport
    VkViewport viewport = {};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = (float) swapConfig.extent.width;
    viewport.height = (float) swapConfig.extent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapConfig.extent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Fill out rasterizer config
    VkPipelineRasterizationStateCreateInfo rasterizerState = {};
    rasterizerState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerState.depthClampEnable = VK_FALSE;
    rasterizerState.rasterizerDiscardEnable = VK_FALSE;
    rasterizerState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerState.lineWidth = 1.f;
    rasterizerState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerState.frontFace = VK_FRONT_FACE_CLOCKWISE; // Clockwise in _screenspace_ with y down
    rasterizerState.depthBiasEnable = VK_FALSE;
    rasterizerState.depthBiasConstantFactor = 0.f; // optional
    rasterizerState.depthBiasClamp = 0.f; // optional
    rasterizerState.depthBiasSlopeFactor = 0.f; // optional

    // Fill out multisampling config
    VkPipelineMultisampleStateCreateInfo multisampleState = {};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleState.minSampleShading = 1.f; // optional
    multisampleState.pSampleMask = nullptr; // optional
    multisampleState.alphaToCoverageEnable = VK_FALSE; // optional
    multisampleState.alphaToOneEnable = VK_FALSE; // optional

    // Fill out blending config
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // optional

    VkPipelineColorBlendStateCreateInfo colorBlendState = {};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.logicOp = VK_LOGIC_OP_COPY; // optional
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachment;
    colorBlendState.blendConstants[0] = 0.f; //optional
    colorBlendState.blendConstants[1] = 0.f; //optional
    colorBlendState.blendConstants[2] = 0.f; //optional
    colorBlendState.blendConstants[3] = 0.f; //optional

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // optional
    pipelineLayoutInfo.pSetLayouts = nullptr; // optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // optional

    if (vkCreatePipelineLayout(_device.handle(), &pipelineLayoutInfo, nullptr, &_vkGraphicsPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    // Create pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizerState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = nullptr; // optional
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = nullptr; // optional
    pipelineInfo.layout = _vkGraphicsPipelineLayout;
    pipelineInfo.renderPass = _vkRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
    pipelineInfo.basePipelineIndex = -1; // optional

    if (vkCreateGraphicsPipelines(_device.handle(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_vkGraphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to crate graphics pipeline");

    vkDestroyShaderModule(_device.handle(), vertShaderModule, nullptr);
    vkDestroyShaderModule(_device.handle(), fragShaderModule, nullptr);
}

void App::createCommandBuffers()
{
    _vkCommandBuffers.resize(_swapchain.imageCount());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = _device.commandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) _vkCommandBuffers.size();

    if (vkAllocateCommandBuffers(_device.handle(), &allocInfo, _vkCommandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");

    for (size_t i = 0; i < _vkCommandBuffers.size(); ++i) {
        // Begin command buffer
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // Might be scheduling next frame while last is rendering
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(_vkCommandBuffers[i], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer");

        // Record renderpass
        VkClearValue clearColor = {0.f, 0.f, 0.f, 0.f};
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = _vkRenderPass;
        renderPassInfo.framebuffer = _swapchain.fbo(i);
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = _swapchain.extent();
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(_vkCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(_vkCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _vkGraphicsPipeline);

        vkCmdDraw(_vkCommandBuffers[i], 3, 1, 0, 0);

        vkCmdEndRenderPass(_vkCommandBuffers[i]);

        if (vkEndCommandBuffer(_vkCommandBuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");
    }
}

void App::createSemaphores()
{
    _imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(_device.handle(), &semaphoreInfo, nullptr, &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_device.handle(), &semaphoreInfo, nullptr, &_renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create semaphores");
    }
}

void App::drawFrame()
{
    size_t currentFrame = _swapchain.currentFrame();
    auto nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[currentFrame]);
    while (!nextImage.has_value()) {
        // Recreate the swap chain as necessary
        recreateSwapchainAndRelated();
        currentFrame = _swapchain.currentFrame();
        nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[currentFrame]);
    }

    // Submit queue
    VkSemaphore waitSemaphores[] = {_imageAvailableSemaphores[currentFrame]};
    VkSemaphore signalSemaphores[] = {_renderFinishedSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_vkCommandBuffers[nextImage.value()];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, _swapchain.currentFence()) != VK_SUCCESS)
        throw std::runtime_error("Failed to submit draw command buffer");

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(1, signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();

}
