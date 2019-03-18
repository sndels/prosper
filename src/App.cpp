#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>

#include "Constants.hpp"
#include "Vertex.hpp"

using namespace glm;

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
            throw  std::runtime_error(std::string{"Failed to open file '"} + filename + "'");

        const auto fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        // Seek to beginning and read
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
        return buffer;
    }

    vk::ShaderModule createShaderModule(const vk::Device device, const std::vector<char>& spv)
    {
        return device.createShaderModule({
            {},
            spv.size(),
            reinterpret_cast<const uint32_t*>(spv.data())
        });
    }

    std::string resPath(const std::string& res)
    {
        return std::string{RES_PATH} + res;
    }
}

App::~App()
{
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
    _device.init(_window.ptr());

    _meshes.push_back(std::make_shared<Mesh>(vertices, indices, &_device));
    _textures.push_back(std::make_shared<Texture>(&_device, resPath("texture/statue.jpg")));

    // TODO: Actual abstraction
    _scene.push_back({_meshes[0], {}, {}, mat4(1.f)});
    _scene.push_back({_meshes[0], {}, {}, mat4(1.f)});

    const SwapchainConfig swapConfig = selectSwapchainConfig(
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
        vec3(0.f, -2.f, -2.f),
        vec3(0.f),
        vec3(0.f, 1.f, 0.f)
    );
    _cam.perspective(
        radians(45.f),
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

    destroySwapchainRelated();
    _swapchain.destroy();

    const SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(),
        _window.height()}
    );

    createRenderPass(swapConfig);
    createGraphicsPipeline(swapConfig);

    _swapchain.create(&_device, _vkRenderPass, swapConfig);

    createCommandBuffers();

    _cam.perspective(
        radians(45.f),
        _window.width() / static_cast<float>(_window.height()),
        0.1f,
        100.f
    );
}

void App::destroySwapchainRelated()
{
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
    const vk::DescriptorSetLayoutBinding cameraLayoutBinding{
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    };
    _vkCameraDescriptorSetLayout = _device.logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &cameraLayoutBinding
    });

    const vk::DescriptorSetLayoutBinding meshInstanceLayoutBinding{
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eVertex
    };
    _vkMeshInstanceDescriptorSetLayout = _device.logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &meshInstanceLayoutBinding
    });

    // Samplers get a separate layout for simplicity
    const vk::DescriptorSetLayoutBinding samplerLayoutBinding{
        0, // binding
        vk::DescriptorType::eCombinedImageSampler,
        1, // descriptorCount
        vk::ShaderStageFlagBits::eFragment
    };
    _vkSamplerDescriptorSetLayout = _device.logical().createDescriptorSetLayout({
        {}, // flags
        1, // bindingCount
        &samplerLayoutBinding
    });
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
    const uint32_t setCount = [&]{
        uint32_t setCount = 0;
        for (auto& size : poolSizes)
            setCount += size.descriptorCount;

        return setCount;
    }();

    _vkDescriptorPool = _device.logical().createDescriptorPool({
        {}, // flags
        setCount, // max sets
        poolSizes.size(),
        poolSizes.data()
    });
}

void App::createDescriptorSets()
{ 
    // Allocate descriptor sets
    const std::vector<vk::DescriptorSetLayout> cameraLayouts(
        _swapchain.imageCount(),
        _vkCameraDescriptorSetLayout
    );
    _vkCameraDescriptorSets = _device.logical().allocateDescriptorSets({
        _vkDescriptorPool,
        static_cast<uint32_t>(cameraLayouts.size()),
        cameraLayouts.data()
    });

    const std::vector<vk::DescriptorSetLayout> meshInstanceLayouts(
        _swapchain.imageCount(),
        _vkMeshInstanceDescriptorSetLayout
    );
    for (auto& meshInstance : _scene)
        meshInstance.descriptorSets = _device.logical().allocateDescriptorSets({
            _vkDescriptorPool,
            static_cast<uint32_t>(meshInstanceLayouts.size()),
            meshInstanceLayouts.data()
        });

    _vkSamplerDescriptorSet = _device.logical().allocateDescriptorSets({
        _vkDescriptorPool,
        1,
        &_vkSamplerDescriptorSetLayout
    })[0];

    // Update them with buffers
    const auto cameraBufferInfos = _cam.bufferInfos();
    for (size_t i = 0; i < _vkCameraDescriptorSets.size(); ++i) {
        const vk::WriteDescriptorSet descriptorWrite{
            _vkCameraDescriptorSets[i],
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            vk::DescriptorType::eUniformBuffer,
            nullptr, // pImageInfo
            &cameraBufferInfos[i]
        };
        _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }

    for (auto& meshInstance : _scene) {
        const auto meshInstanceBufferInfos = meshInstance.bufferInfos();
        for (size_t i = 0; i < meshInstance.descriptorSets.size(); ++i) {
            const vk::WriteDescriptorSet descriptorWrite{
                meshInstance.descriptorSets[i],
                0, // dstBinding,
                0, // dstArrayElement
                1, // descriptorCount
                vk::DescriptorType::eUniformBuffer,
                nullptr, // pImageInfo
                &meshInstanceBufferInfos[i]
            };
            _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }
    }

    {
        const auto imageInfo = _textures[0]->imageInfo();
        const vk::WriteDescriptorSet descriptorWrite{
            _vkSamplerDescriptorSet,
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            vk::DescriptorType::eCombinedImageSampler,
            &imageInfo
        };
        _device.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }

}

void App::createRenderPass(const SwapchainConfig& swapConfig)
{
    const vk::AttachmentDescription swapAttachment{
        {}, // flags
        swapConfig.surfaceFormat.format,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, // loadOp
        vk::AttachmentStoreOp::eStore, // storeOp
        vk::AttachmentLoadOp::eDontCare, // stencilLoadOp
        vk::AttachmentStoreOp::eDontCare, // stencilStoreOp
        vk::ImageLayout::eUndefined, // initialLayout
        vk::ImageLayout::ePresentSrcKHR // finalLayout
    };
    const vk::AttachmentReference swapAttachmentRef{
        0, // attachment
        vk::ImageLayout::eColorAttachmentOptimal
    };

    // Output
    const vk::SubpassDescription subpass{
        {}, // flags
        vk::PipelineBindPoint::eGraphics,
        0, // inputAttachmentCount
        nullptr, // pInputAttachments
        1, // colorAttachmentCount
        &swapAttachmentRef
    };

    // Synchronize
    const vk::SubpassDependency dependency{
        VK_SUBPASS_EXTERNAL, // srcSubpass
        0, // dstSubpass
        vk::PipelineStageFlagBits::eColorAttachmentOutput, // srcStageMask
        vk::PipelineStageFlagBits::eColorAttachmentOutput, // dstStageMask
        {}, // srcAccessMask
        vk::AccessFlagBits::eColorAttachmentRead |
        vk::AccessFlagBits::eColorAttachmentWrite // dstAccessMask
    };

    _vkRenderPass = _device.logical().createRenderPass({
        {}, // flags
        1, // attachmentCount
        &swapAttachment,
        1, // subpassCount
        &subpass,
        1, // dependencyCount
        &dependency
    });
}

void App::createGraphicsPipeline(const SwapchainConfig& swapConfig)
{
    const auto vertSPV = readFile("shader/shader.vert.spv");
    const auto fragSPV = readFile("shader/shader.frag.spv");
    const vk::ShaderModule vertShaderModule = createShaderModule(_device.logical(), vertSPV);
    const vk::ShaderModule fragShaderModule = createShaderModule(_device.logical(), fragSPV);
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {{
        {
            {}, // flags
            vk::ShaderStageFlagBits::eVertex,
            vertShaderModule,
            "main"
        },
        {
            {}, // flags
            vk::ShaderStageFlagBits::eFragment,
            fragShaderModule,
            "main"
        }
    }};

    const auto vertexBindingDescription = Vertex::bindingDescription();
    const auto vertexAttributeDescriptions = Vertex::attributeDescriptions();
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
        {}, // flags
        1, // vertexBindingDescriptionCount
        &vertexBindingDescription,
        vertexAttributeDescriptions.size(),
        vertexAttributeDescriptions.data()
    };

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        {}, // flags
        vk::PrimitiveTopology::eTriangleList,
        VK_FALSE // primitiveRestartEnable
    };

    const vk::Viewport viewport{
        0.f, // x
        0.f, // y
        static_cast<float>(swapConfig.extent.width), // width
        static_cast<float>(swapConfig.extent.height), // height
        0.f, // minDepth
        1.f // maxDepth
    };
    const vk::Rect2D scissor{
        {0, 0}, // offset
        swapConfig.extent
    };
    const vk::PipelineViewportStateCreateInfo viewportState{
        {}, // flags
        1, // viewportCount
        &viewport,
        1, // scissorCount
        &scissor
    };

    const vk::PipelineRasterizationStateCreateInfo rasterizerState{
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
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        {}, //flags
        vk::SampleCountFlagBits::e1 // rasterationSamples
    };
    // TODO: sampleshading now 0, verify it doesn't matter if not enabled

    const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
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
    };
    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        {}, // flags
        VK_FALSE, // logicOpEnable
        vk::LogicOp::eCopy,
        1, // attachmentCount
        &colorBlendAttachment
    };

    const std::array<vk::DescriptorSetLayout, 3> setLayouts = {{
        _vkCameraDescriptorSetLayout,
        _vkMeshInstanceDescriptorSetLayout,
        _vkSamplerDescriptorSetLayout
    }};
    _vkGraphicsPipelineLayout = _device.logical().createPipelineLayout({
        {}, // flags
        setLayouts.size(),
        setLayouts.data()
    });

    _vkGraphicsPipeline = _device.logical().createGraphicsPipeline({}, {
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
    });

    _device.logical().destroyShaderModule(vertShaderModule);
    _device.logical().destroyShaderModule(fragShaderModule);
}

void App::createCommandBuffers()
{
    _vkCommandBuffers = _device.logical().allocateCommandBuffers({
        _device.commandPool(),
        vk::CommandBufferLevel::ePrimary,
        _swapchain.imageCount() // commandBufferCount
    });
}

void App::createSemaphores()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        _imageAvailableSemaphores.push_back(_device.logical().createSemaphore({}));
        _renderFinishedSemaphores.push_back(_device.logical().createSemaphore({}));
    }
}

void App::drawFrame()
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t currentFrame = _swapchain.currentFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&]{
        auto nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[currentFrame]);
        while (!nextImage.has_value()) {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[currentFrame]);
        }

        return nextImage.value();
    }();

    updateUniformBuffers(nextImage);

    recordCommandBuffer(nextImage);

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
    const vk::SubmitInfo submitInfo{
        waitSemaphores.size(),
        waitSemaphores.data(),
        waitStages.data(),
        1, // commandBufferCount
        &_vkCommandBuffers[nextImage],
        signalSemaphores.size(),
        signalSemaphores.data()
    };
    _device.graphicsQueue().submit(1, &submitInfo, _swapchain.currentFence());

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores.size(), signalSemaphores.data()) ||
        _window.resized())
        recreateSwapchainAndRelated();

}

void App::updateUniformBuffers(const uint32_t nextImage)
{
    _cam.updateBuffer(nextImage);

    static const auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    for (size_t i = 0; i < _scene.size(); ++i) {
        _scene[i].modelToWorld = rotate(
                                   translate(
                                       mat4(1.f),
                                       vec3(-1.f + 2.f * i, 0.f, 0.f)
                                   ),
                                   (-1.f + 2.f * i) * time * radians(360.f),
                                   vec3(0.f, 0.f, 1.f)
                                 );
        _scene[i].updateBuffer(&_device, nextImage);
    }
}

void App::recordCommandBuffer(const uint32_t nextImage)
{
    const vk::CommandBuffer buffer = _vkCommandBuffers[nextImage];
    buffer.reset({});

    buffer.begin({
        vk::CommandBufferUsageFlagBits::eSimultaneousUse
    });

    const vk::ClearValue clearColor(std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
    buffer.beginRenderPass(
        {
            _vkRenderPass,
            _swapchain.fbo(nextImage),
            vk::Rect2D{
                {0, 0}, // offset
                _swapchain.extent()
            },
            1, // clearValueCount
            &clearColor
        },
        vk::SubpassContents::eInline
    );

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

    buffer.endRenderPass();
    buffer.end();
}
