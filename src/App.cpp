#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>

#include "Constants.hpp"
#include "Vertex.hpp"

namespace {
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, 0.f}, {1.f, 0.f, 0.f}},
        {{ 0.5f, -0.5f, 0.f}, {0.f, 1.f, 0.f}},
        {{ 0.5f,  0.5f, 0.f}, {0.f, 0.f, 1.f}},
        {{-0.5f,  0.5f, 0.f}, {1.f, 1.f, 1.f}}
    };
    const std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0
    };

    static std::vector<char> readFile(const std::string& filename)
    {
        // Open from end to find size from initial position
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw  std::runtime_error(std::string("Failed to open file '") + filename + "'");

        const auto fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        // Seek to beginning and read
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }

    vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& spv)
    {
        const vk::ShaderModuleCreateInfo createInfo(
            {},
            spv.size(),
            reinterpret_cast<const uint32_t*>(spv.data())
        );
        return device.createShaderModule(createInfo);
    }
}

App::~App()
{
    // Destroy vulkan stuff
    for (auto& semaphore : _renderFinishedSemaphores)
        _device.logical().destroySemaphore(semaphore);
    for (auto& semaphore : _imageAvailableSemaphores)
        _device.logical().destroySemaphore(semaphore);

    destroySwapchainRelated();

    _device.logical().destroyDescriptorPool(_vkDescriptorPool);

    for (auto& buffer : _transformBuffers) {
        _device.logical().destroyBuffer(buffer.handle);
        _device.logical().freeMemory(buffer.memory);
    }

    _device.logical().destroyDescriptorSetLayout(_vkCameraDescriptorSetLayout);
}

void App::init()
{
    _window.init(WIDTH, HEIGHT, "prosper");

    // Init vulkan
    _device.init(_window.ptr());

    _meshes.emplace_back(vertices, indices, &_device);

    SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(), _window.height()}
    );

    createRenderPass(swapConfig);

    createDescriptorSetLayout();
    createGraphicsPipeline(swapConfig);

    _swapchain.create(&_device, _vkRenderPass, swapConfig);

    _cam.createUniformBuffers(&_device, _swapchain.imageCount());
    createUniformBuffers();
    createDescriptorPool();

    createDescriptorSets();

    createCommandBuffers();

    createSemaphores();

    _cam.lookAt(
        glm::vec3(0.f, -2.f, -2.f),
        glm::vec3(0.f),
        glm::vec3(0.f, 1.f, 0.f)
    );
    _cam.perspective(
        glm::radians(45.f),
        _window.width() / static_cast<float>(_window.height()),
        0.1f,
        100.f
    );
}

void App::run() 
{
    while (_window.open()) {
        _window.startFrame();
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    _device.logical().waitIdle();
}

void App::recreateSwapchainAndRelated()
{
    while (_window.width() == 0 && _window.height() == 0) {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device.logical().waitIdle();

    // Destroy resources tied to current swapchain
    destroySwapchainRelated();

    // Don't forget the actual swapchain
    _swapchain.destroy();

    // Create new swapchain and tied resources
    SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(),
        _window.height()}
    );

    createRenderPass(swapConfig);
    createGraphicsPipeline(swapConfig);

    _swapchain.create(&_device, _vkRenderPass, swapConfig);

    createCommandBuffers();

    // Update camera
    _cam.perspective(
        glm::radians(45.f),
        _window.width() / static_cast<float>(_window.height()),
        0.1f,
        100.f
    );
}

void App::destroySwapchainRelated()
{
    // Destroy related vulkan resources
    _device.logical().freeCommandBuffers(
        _device.commandPool(),
        _vkCommandBuffers.size(),
        _vkCommandBuffers.data()
    );
    _device.logical().destroyPipeline(_vkGraphicsPipeline);
    _device.logical().destroyPipelineLayout(_vkGraphicsPipelineLayout);
    _device.logical().destroyRenderPass(_vkRenderPass);
}

void App::createDescriptorSetLayout()
{
    // Create binding for Camera
    const vk::DescriptorSetLayoutBinding cameraLayoutBinding(
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptor count
        vk::ShaderStageFlagBits::eVertex
    );

    // Create descriptor set layout
    const vk::DescriptorSetLayoutCreateInfo layoutInfo(
        {}, // flags
        1, // binding count
        &cameraLayoutBinding
    );
    _vkCameraDescriptorSetLayout = _device.logical().createDescriptorSetLayout(layoutInfo);

    // TODO: Object descriptor set layout
}

void App::createUniformBuffers()
{
    const vk::DeviceSize bufferSize = sizeof(Transforms);

    for (size_t i = 0; i < _swapchain.imageCount(); ++i)
        _transformBuffers.push_back(_device.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent
        ));
}

void App::createDescriptorPool()
{
    const vk::DescriptorPoolSize poolSize(
        vk::DescriptorType::eUniformBuffer,
        _swapchain.imageCount() // descriptor count
    );

    const vk::DescriptorPoolCreateInfo poolInfo(
        {}, // flags
        _swapchain.imageCount(), // max sets
        1, // poolsize count
        &poolSize
    );

    _vkDescriptorPool = _device.logical().createDescriptorPool(poolInfo);
}

void App::createDescriptorSets()
{ 
    // Allocate descriptor sets
    const std::vector<vk::DescriptorSetLayout> layouts(
        _swapchain.imageCount(),
        _vkCameraDescriptorSetLayout
    );
    const vk::DescriptorSetAllocateInfo allocInfo(
        _vkDescriptorPool,
        layouts.size(),
        layouts.data()
    );
    _vkCameraDescriptorSets = _device.logical().allocateDescriptorSets(allocInfo);
    // TODO: Object descriptor sets

    // Update them with buffers
    auto cameraBufferInfos = _cam.bufferInfos();
    for (size_t i = 0; i < _vkCameraDescriptorSets.size(); ++i) {
        const vk::WriteDescriptorSet descriptorWrite(
            _vkCameraDescriptorSets[i],
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            vk::DescriptorType::eUniformBuffer,
            nullptr, // pImageInfo
            &cameraBufferInfos[i]
        );
        _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        // TODO: Object descriptor sets
    }
}

void App::createRenderPass(const SwapchainConfig& swapConfig)
{
    // Fill color attachment data for the swap buffer
    const vk::AttachmentDescription colorAttachment(
        {}, // flags
        swapConfig.surfaceFormat.format,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, // loadOp
        vk::AttachmentStoreOp::eStore, // storeOp
        vk::AttachmentLoadOp::eDontCare, // stencilLoadOp
        vk::AttachmentStoreOp::eDontCare, // stencilStoreOp
        vk::ImageLayout::eUndefined, // initialLayout
        vk::ImageLayout::ePresentSrcKHR // finalLayout
    );
    const vk::AttachmentReference colorAttachmentRef(
        0, // attachment
        vk::ImageLayout::eColorAttachmentOptimal
    );

    // Create subpass for output
    const vk::SubpassDescription subpass(
        {}, // flags
        vk::PipelineBindPoint::eGraphics,
        0, // inputAttachmentCount
        nullptr, // pInputAttachments
        1, // colorAttachmentCount
        &colorAttachmentRef
    );

    // Create subpass dependency to synchronize
    const vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, // srcSubpass
        0, // dstSubpass
        vk::PipelineStageFlagBits::eColorAttachmentOutput, // srcStageMask
        vk::PipelineStageFlagBits::eColorAttachmentOutput, // dstStageMask
        {}, // srcAccessMask
        vk::AccessFlagBits::eColorAttachmentRead |
        vk::AccessFlagBits::eColorAttachmentWrite // dstAccessMask
    );

    // Create render pass
    const vk::RenderPassCreateInfo renderPassInfo(
        {}, // flags
        1, // attachmentCount
        &colorAttachment,
        1, // subpassCount
        &subpass,
        1, // dependencyCount
        &dependency
    );
    _vkRenderPass = _device.logical().createRenderPass(renderPassInfo);
}

void App::createGraphicsPipeline(const SwapchainConfig& swapConfig)
{
    // Create modules for shaders
    const auto vertSPV = readFile("shader/shader.vert.spv");
    const auto fragSPV = readFile("shader/shader.frag.spv");
    vk::ShaderModule vertShaderModule = createShaderModule(_device.logical(), vertSPV);
    vk::ShaderModule fragShaderModule = createShaderModule(_device.logical(), fragSPV);

    // Fill out create infos for the shader stages
    const vk::PipelineShaderStageCreateInfo vertStageInfo(
        {}, // flags
        vk::ShaderStageFlagBits::eVertex,
        vertShaderModule,
        "main"
    );
    const vk::PipelineShaderStageCreateInfo fragStageInfo(
        {}, // flags
        vk::ShaderStageFlagBits::eFragment,
        fragShaderModule,
        "main"
    );
    const vk::PipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    // Config shader stage inputs
    const auto vertexBindingDescription = Vertex::bindingDescription();
    const auto vertexAttributeDescriptions = Vertex::attributeDescriptions();
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo(
        {}, // flags
        1, // vertexBindingDescriptionCount
        &vertexBindingDescription,
        vertexAttributeDescriptions.size(),
        vertexAttributeDescriptions.data()
    );

    // Config input topology
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, // flags
        vk::PrimitiveTopology::eTriangleList,
        VK_FALSE // primitiveRestartEnable
    );

    // Set up viewport
    const vk::Viewport viewport(
        0.f, // x
        0.f, // y
        static_cast<float>(swapConfig.extent.width), // width
        static_cast<float>(swapConfig.extent.height), // height
        0.f, // minDepth
        1.f // maxDepth
    );
    const vk::Rect2D scissor(
        {0, 0}, // offset
        swapConfig.extent
    );
    const vk::PipelineViewportStateCreateInfo viewportState(
        {}, // flags
        1, // viewportCount
        &viewport,
        1, // scissorCount
        &scissor
    );

    // Config rasterizer
    const vk::PipelineRasterizationStateCreateInfo rasterizerState(
        {}, // flags
        VK_FALSE, // depthClampEnable
        VK_FALSE, // rasterizerDiscardEnable
        vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack,
        vk::FrontFace::eCounterClockwise,
        VK_FALSE, // depthBiasEnable
        0.f, // depthBiasConstantFactor
        0.f, // depthBiasClamp
        0.f, // depthBiasSlopeOperator
        1.f // lineWidth
    );

    // Config multisampling
    const vk::PipelineMultisampleStateCreateInfo multisampleState(
        {}, //flags
        vk::SampleCountFlagBits::e1 // rasterationSamples
    );
    // TODO: sampleshading now 0, verify it doesn't matter if not enabled

    // Config blending
    // TODO: is default enough?
    const vk::PipelineColorBlendAttachmentState colorBlendAttachment(
        VK_FALSE, // blendEnable
        vk::BlendFactor::eOne, // srcColorBlendFactor
        vk::BlendFactor::eZero, // dstColorBlendFactor
        vk::BlendOp::eAdd, // colorBlendOp
        vk::BlendFactor::eOne, // srcAlphaBlendFactor
        vk::BlendFactor::eZero, // dstAlphaBlendFactor
        vk::BlendOp::eAdd, // alphaBlendOp
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA // colorWriteMas
    );
    const vk::PipelineColorBlendStateCreateInfo colorBlendState(
        {}, // flags
        VK_FALSE, // logicOpEnable
        vk::LogicOp::eCopy,
        1, // attachmentCount
        &colorBlendAttachment
    );

    // Create pipeline layout
    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
        {}, // flags
        1, // setLayoutCount
        &_vkCameraDescriptorSetLayout
    );
    _vkGraphicsPipelineLayout = _device.logical().createPipelineLayout(pipelineLayoutInfo);

    // Create pipeline
    const vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, // flags
        sizeof(shaderStages) / sizeof(vk::PipelineShaderStageCreateInfo), // stageCount
        shaderStages,
        &vertInputInfo,
        &inputAssembly,
        nullptr, // tessellation
        &viewportState,
        &rasterizerState,
        &multisampleState,
        nullptr, // depth stencil
        &colorBlendState,
        nullptr, // dynamic
        _vkGraphicsPipelineLayout,
        _vkRenderPass,
        0 // subpass
    );
    _vkGraphicsPipeline = _device.logical().createGraphicsPipeline({}, pipelineInfo);

    // Clean up shaders
    _device.logical().destroyShaderModule(vertShaderModule);
    _device.logical().destroyShaderModule(fragShaderModule);
}

void App::createCommandBuffers()
{
    const vk::CommandBufferAllocateInfo allocInfo(
        _device.commandPool(),
        vk::CommandBufferLevel::ePrimary,
        _swapchain.imageCount() // commandBufferCount
    );
    _vkCommandBuffers = _device.logical().allocateCommandBuffers(allocInfo);
}

void App::createSemaphores()
{
    const vk::SemaphoreCreateInfo semaphoreInfo;
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        _imageAvailableSemaphores.push_back(_device.logical().createSemaphore(semaphoreInfo));
        _renderFinishedSemaphores.push_back(_device.logical().createSemaphore(semaphoreInfo));
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

    // Update uniform buffers
    _cam.updateBuffer(nextImage.value());
    updateUniformBuffer(nextImage.value());

    // Record frame
    recordCommandBuffer(nextImage.value());

    // Submit queue
    const vk::Semaphore waitSemaphores[] = {_imageAvailableSemaphores[currentFrame]};
    const vk::Semaphore signalSemaphores[] = {_renderFinishedSemaphores[currentFrame]};
    const vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const vk::SubmitInfo submitInfo(
        sizeof(waitSemaphores) / sizeof(vk::Semaphore), // waitSemaphoreCount
        waitSemaphores,
        waitStages,
        1, // commandBufferCount
        &_vkCommandBuffers[nextImage.value()],
        sizeof(signalSemaphores) / sizeof(vk::Semaphore), // signalSemaphoreCount
        signalSemaphores
    );
    _device.graphicsQueue().submit(1, &submitInfo, _swapchain.currentFence());

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(1, signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();

}

void App::updateUniformBuffer(uint32_t nextImage)
{
    // TODO: object uniform buffers
    /*
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    Transforms transforms;
    transforms.modelToClip = glm::rotate(
                                 glm::mat4(1.f),
                                 time * glm::radians(360.f),
                                 glm::vec3(0.f, 0.f, 1.f)
                             );

    void* data;
    _device.logical().mapMemory(_transformBuffers[nextImage].memory, 0, sizeof(Transforms), {}, &data);
    memcpy(data, &transforms, sizeof(Transforms));
    _device.logical().unmapMemory(_transformBuffers[nextImage].memory);
    */
}

void App::recordCommandBuffer(uint32_t nextImage)
{
    vk::CommandBuffer buffer = _vkCommandBuffers[nextImage];
    // Reset command buffer
    buffer.reset({});

    // Begin command buffer
    vk::CommandBufferBeginInfo beginInfo(
        vk::CommandBufferUsageFlagBits::eSimultaneousUse
    );
    buffer.begin(beginInfo);

    // Record renderpass
    vk::ClearValue clearColor(std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
    vk::RenderPassBeginInfo renderPassInfo(
        _vkRenderPass,
        _swapchain.fbo(nextImage),
        vk::Rect2D(
            {0, 0}, // offset
            _swapchain.extent()
        ),
        1, // clearValueCount
        &clearColor
    );

    // Begin
    buffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    // Bind pipeline
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _vkGraphicsPipeline);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        _vkGraphicsPipelineLayout,
        0, // firstSet
        1, // descriptorSetCount
        &_vkCameraDescriptorSets[nextImage],
        0, // dynamicOffsetCount
        nullptr // pDynamicOffsets
    );

    // Draw meshes
    // TODO: Object descriptor sets
    for (auto& mesh : _meshes)
        mesh.draw(buffer);

    // End
    buffer.endRenderPass();
    buffer.end();
}
