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
        {{-0.5f, -0.5f, 0.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}},
        {{ 0.5f, -0.5f, 0.f}, {1.f, 1.f}, {0.f, 1.f, 0.f}},
        {{ 0.5f,  0.5f, 0.f}, {1.f, 0.f}, {0.f, 0.f, 1.f}},
        {{-0.5f,  0.5f, 0.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}}
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

    std::string resPath(const std::string& res)
    {
        return std::string(RES_PATH) + res;
    }
}

App::~App()
{
    // Destroy vulkan stuff
    for (auto& semaphore : _renderFinishedSemaphores)
        _device.logical().destroy(semaphore);
    for (auto& semaphore : _imageAvailableSemaphores)
        _device.logical().destroy(semaphore);

    destroySwapchainRelated();

    _device.logical().destroy(_vkDescriptorPool);

    for (auto& instance : _scene) {
        for (auto& buffer : instance.uniformBuffers) {
            _device.logical().destroy(buffer.handle);
            _device.logical().free(buffer.memory);
        }
    }

    _device.logical().destroy(_vkCameraDescriptorSetLayout);
    _device.logical().destroy(_vkMeshInstanceDescriptorSetLayout);
    _device.logical().destroy(_vkSamplerDescriptorSetLayout);
}

void App::init()
{
    _window.init(WIDTH, HEIGHT, "prosper");

    // Init vulkan
    _device.init(_window.ptr());

    _meshes.push_back(std::make_shared<Mesh>(vertices, indices, &_device));
    _textures.push_back(std::make_shared<Texture>(&_device, resPath("texture/statue.jpg")));

    // TODO: Actual abstraction
    _scene.push_back(MeshInstance{_meshes[0], {}, {}, glm::mat4(1.f)});
    _scene.push_back(MeshInstance{_meshes[0], {}, {}, glm::mat4(1.f)});

    SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(), _window.height()}
    );

    createRenderPass(swapConfig);

    createDescriptorSetLayouts();
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
    _device.logical().destroy(_vkGraphicsPipeline);
    _device.logical().destroy(_vkGraphicsPipelineLayout);
    _device.logical().destroy(_vkRenderPass);
}

void App::createDescriptorSetLayouts()
{
    // Separate bindings for camera and mesh instance to avoid having camera bound to all mesh instance sets
    const vk::DescriptorSetLayoutBinding cameraLayoutBinding(
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    );
    const vk::DescriptorSetLayoutCreateInfo cameraLayoutInfo(
        {}, // flags
        1, // bindingCount
        &cameraLayoutBinding
    );
    _vkCameraDescriptorSetLayout = _device.logical().createDescriptorSetLayout(cameraLayoutInfo);

    const vk::DescriptorSetLayoutBinding meshInstanceLayoutBinding(
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    );
    const vk::DescriptorSetLayoutCreateInfo meshInstanceLayoutInfo(
        {}, // flags
        1, // bindingCount
        &meshInstanceLayoutBinding
    );
    _vkMeshInstanceDescriptorSetLayout = _device.logical().createDescriptorSetLayout(meshInstanceLayoutInfo);

    // Samplers get a separate layout for simplicity
    const vk::DescriptorSetLayoutBinding samplerLayoutBinding(
        0, // binding
        vk::DescriptorType::eCombinedImageSampler,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eFragment
    );
    const vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo (
        {}, // flags
        1, // bindingCount
        &samplerLayoutBinding
    );
    _vkSamplerDescriptorSetLayout = _device.logical().createDescriptorSetLayout(samplerLayoutInfo);
}

void App::createUniformBuffers()
{
    // TODO: Abstract mesh instances
    const vk::DeviceSize bufferSize = sizeof(MeshInstanceUniforms);

    for (auto& meshInstance : _scene) {
        for (size_t i = 0; i < _swapchain.imageCount(); ++i)
            meshInstance.uniformBuffers.push_back(_device.createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            ));
    }
}

void App::createDescriptorPool()
{
    const std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
        {
            vk::DescriptorType::eUniformBuffer,
            _swapchain.imageCount() * (1 + static_cast<uint32_t>(_scene.size())) // descriptor count, camera and mesh instances
        },
        {
            vk::DescriptorType::eCombinedImageSampler,
            _swapchain.imageCount() // descriptor count
        }
    }};
    uint32_t setCount = 0;
    for (auto& size : poolSizes)
        setCount += size.descriptorCount;

    const vk::DescriptorPoolCreateInfo poolInfo(
        {}, // flags
        setCount, // max sets
        poolSizes.size(),
        poolSizes.data()
    );

    _vkDescriptorPool = _device.logical().createDescriptorPool(poolInfo);
}

void App::createDescriptorSets()
{ 
    // Allocate descriptor sets
    const std::vector<vk::DescriptorSetLayout> cameraLayouts(
        _swapchain.imageCount(),
        _vkCameraDescriptorSetLayout
    );
    const vk::DescriptorSetAllocateInfo cameraAllocInfo(
        _vkDescriptorPool,
        cameraLayouts.size(),
        cameraLayouts.data()
    );
    _vkCameraDescriptorSets = _device.logical().allocateDescriptorSets(cameraAllocInfo);

    const std::vector<vk::DescriptorSetLayout> meshInstanceLayouts(
        _swapchain.imageCount(),
        _vkMeshInstanceDescriptorSetLayout
    );
    const vk::DescriptorSetAllocateInfo meshInstanceAllocInfo(
        _vkDescriptorPool,
        meshInstanceLayouts.size(),
        meshInstanceLayouts.data()
    );
    for (auto& meshInstance : _scene)
        meshInstance.descriptorSets = _device.logical().allocateDescriptorSets(meshInstanceAllocInfo);

    const vk::DescriptorSetAllocateInfo samplerAllocInfo(
        _vkDescriptorPool,
        1,
        &_vkSamplerDescriptorSetLayout
    );
    _vkSamplerDescriptorSet = _device.logical().allocateDescriptorSets(samplerAllocInfo)[0];


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
    }

    for (auto& meshInstance : _scene) {
        auto meshInstanceBufferInfos = meshInstance.bufferInfos();
        for (size_t i = 0; i < meshInstance.descriptorSets.size(); ++i) {
            const vk::WriteDescriptorSet descriptorWrite(
                meshInstance.descriptorSets[i],
                0, // dstBinding,
                0, // dstArrayElement
                1, // descriptorCount
                vk::DescriptorType::eUniformBuffer,
                nullptr, // pImageInfo
                &meshInstanceBufferInfos[i]
            );
            _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }
    }

    {
        const auto imageInfo = _textures[0]->imageInfo();
        const vk::WriteDescriptorSet descriptorWrite(
            _vkSamplerDescriptorSet,
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            vk::DescriptorType::eCombinedImageSampler,
            &imageInfo
        );
        _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
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
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {{vertStageInfo, fragStageInfo}};

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
    const std::array<vk::DescriptorSetLayout, 3> setLayouts = {{
        _vkCameraDescriptorSetLayout,
        _vkMeshInstanceDescriptorSetLayout,
        _vkSamplerDescriptorSetLayout
    }};
    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
        {}, // flags
        setLayouts.size(),
        setLayouts.data()
    );
    _vkGraphicsPipelineLayout = _device.logical().createPipelineLayout(pipelineLayoutInfo);

    // Create pipeline
    const vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, // flags
        shaderStages.size(),
        shaderStages.data(),
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
    updateUniformBuffers(nextImage.value());

    // Record frame
    recordCommandBuffer(nextImage.value());

    // Submit queue
    const std::array<vk::Semaphore, 1> waitSemaphores = {{
        _imageAvailableSemaphores[currentFrame]
    }};
    const std::array<vk::PipelineStageFlags, 1> waitStages = {{
        vk::PipelineStageFlagBits::eColorAttachmentOutput
    }};
    const std::array<vk::Semaphore, 1> signalSemaphores = {{
        _renderFinishedSemaphores[currentFrame]
    }};
    const vk::SubmitInfo submitInfo(
        waitSemaphores.size(),
        waitSemaphores.data(),
        waitStages.data(),
        1, // commandBufferCount
        &_vkCommandBuffers[nextImage.value()],
        signalSemaphores.size(),
        signalSemaphores.data()
    );
    _device.graphicsQueue().submit(1, &submitInfo, _swapchain.currentFence());

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores.size(), signalSemaphores.data()) ||
        _window.resized())
        recreateSwapchainAndRelated();

}

void App::updateUniformBuffers(uint32_t nextImage)
{
    _cam.updateBuffer(nextImage);

    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    for (size_t i = 0; i < _scene.size(); ++i) {
        _scene[i].modelToWorld = glm::rotate(
                                      glm::translate(
                                          glm::mat4(1.f),
                                          glm::vec3(-1.f + 2.f * i, 0.f, 0.f)
                                      ),
                                      (-1.f + 2.f * i) * time * glm::radians(360.f),
                                      glm::vec3(0.f, 0.f, 1.f)
                                  );
        _scene[i].updateBuffer(&_device, nextImage);
    }
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

    // Draw scene
    for (auto& meshInstance : _scene) {
        const std::array<vk::DescriptorSet, 2> sets = {{
            meshInstance.descriptorSets[nextImage],
            _vkSamplerDescriptorSet
        }};
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            _vkGraphicsPipelineLayout,
            1, // firstSet
            sets.size(),
            sets.data(),
            0, // dynamicOffsetCount
            nullptr // pDynamicOffsets
        );
        meshInstance.mesh->draw(buffer);
    }

    // End
    buffer.endRenderPass();
    buffer.end();
}
