#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>

#include "Constants.hpp"
#include "InputHandler.hpp"
#include "Vertex.hpp"

using namespace glm;

using std::cout;
using std::cerr;
using std::endl;

namespace {
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    const float CAMERA_FOV = 59.f;
    const float CAMERA_NEAR = 0.001f;
    const float CAMERA_FAR = 512.f;

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
}

App::~App()
{
    destroySwapchainRelated();

    for (auto& semaphore : _renderFinishedSemaphores)
        _device.logical().destroy(semaphore);
    for (auto& semaphore : _imageAvailableSemaphores)
        _device.logical().destroy(semaphore);

    _device.logical().destroy(_vkDescriptorPool);
}

void App::init()
{
    _window.init(WIDTH, HEIGHT, "prosper");
    _device.init(_window.ptr());

    const SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(), _window.height()}
    );

    // Resources tied to specific swap images via command buffers
    createDescriptorPool(swapConfig.imageCount);

    _cam.createUniformBuffers(&_device, swapConfig.imageCount);
    _cam.createDescriptorSets(
        _vkDescriptorPool,
        swapConfig.imageCount,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
    );

    _world.loadGLTF(
        &_device,
        swapConfig.imageCount,
        resPath("glTF/FlightHelmet/glTF/FlightHelmet.gltf")
    );

    // Semaphores are correspond to logical frames instead of swapchain images
    createSemaphores(MAX_FRAMES_IN_FLIGHT);

    createRenderPass(swapConfig);
    createGraphicsPipeline(swapConfig);
    _swapchain.create(&_device, _vkRenderPass, swapConfig);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);

    _cam.lookAt(
        vec3{0.25f, 0.2f, 0.75f},
        vec3{0.f},
        vec3{0.f, 1.f, 0.f}
    );
    _cam.perspective(
        radians(CAMERA_FOV),
        _window.width() / static_cast<float>(_window.height()),
        CAMERA_NEAR,
        CAMERA_FAR
    );
}

void App::run() 
{
    while (_window.open()) {
        _window.startFrame();

        const auto& mouse = InputHandler::instance().mouse();
        if (mouse.leftDown && mouse.currentPos != mouse.lastPos) {
            _cam.orbit(
                mouse.currentPos,
                mouse.lastPos,
                vec2(_window.width(), _window.height()) / 2.f
            );
        }
        if (mouse.rightDown && mouse.currentPos != mouse.lastPos) {
            _cam.scaleOrbit(
                mouse.currentPos.y,
                 mouse.lastPos.y,
                _window.height() / 2.f
            );
        }
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
    createCommandBuffers(swapConfig);

    _cam.perspective(
        radians(CAMERA_FOV),
        _window.width() / static_cast<float>(_window.height()),
        CAMERA_NEAR,
        CAMERA_FAR
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

void App::createDescriptorPool(const uint32_t swapImageCount)
{
    const vk::DescriptorPoolSize poolSize{
            vk::DescriptorType::eUniformBuffer,
            swapImageCount // descriptorCount, camera and mesh instances
        };
    _vkDescriptorPool = _device.logical().createDescriptorPool({
        {}, // flags
        poolSize.descriptorCount, // max sets
        1,
        &poolSize
    });
}

void App::createSemaphores(const uint32_t concurrentFrameCount)
{
    for (size_t i = 0; i < concurrentFrameCount; ++i) {
        _imageAvailableSemaphores.push_back(_device.logical().createSemaphore({}));
        _renderFinishedSemaphores.push_back(_device.logical().createSemaphore({}));
    }
}

void App::createRenderPass(const SwapchainConfig& swapConfig)
{
    // TODO: Can swap surface formats change after first creation?
    const std::array<vk::AttachmentDescription, 2> attachments = {{
        { // swap color
            {}, // flags
            swapConfig.surfaceFormat.format,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear, // loadOp
            vk::AttachmentStoreOp::eStore, // storeOp
            vk::AttachmentLoadOp::eDontCare, // stencilLoadOp
            vk::AttachmentStoreOp::eDontCare, // stencilStoreOp
            vk::ImageLayout::eUndefined, // initialLayout
            vk::ImageLayout::ePresentSrcKHR // finalLayout
        },
        { // depth
            {}, // flags
            swapConfig.depthFormat,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear, // loadOp
            vk::AttachmentStoreOp::eDontCare, // storeOp
            vk::AttachmentLoadOp::eDontCare, // stencilLoadOp
            vk::AttachmentStoreOp::eDontCare, // stencilStoreOp
            vk::ImageLayout::eUndefined, // initialLayout
            vk::ImageLayout::eDepthStencilAttachmentOptimal // finalLayout
        }
    }};
    const vk::AttachmentReference swapAttachmentRef{
        0, // attachment
        vk::ImageLayout::eColorAttachmentOptimal
    };
    const vk::AttachmentReference depthAttachmentRef{
        1, // attachment
        vk::ImageLayout::eDepthStencilAttachmentOptimal
    };

    // Output
    const vk::SubpassDescription subpass{
        {}, // flags
        vk::PipelineBindPoint::eGraphics,
        0, // inputAttachmentCount
        nullptr, // pInputAttachments
        1, // colorAttachmentCount
        &swapAttachmentRef,
        nullptr, // pResolveAttachments
        &depthAttachmentRef
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
        attachments.size(),
        attachments.data(),
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

    // TODO: Dynamic viewport state?
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

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        {}, // flags
        VK_TRUE, // depthTestEnable
        VK_TRUE, // depthWriteEnable
        vk::CompareOp::eLess
    };

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
        _cam.descriptorSetLayout(),
        _world._modelInstanceDSLayout,
        _world._materialDSLayout
    }};
    const vk::PushConstantRange pcRange{
        vk::ShaderStageFlagBits::eFragment,
        0, // offset
        sizeof(Material::PCBlock)
    };
    _vkGraphicsPipelineLayout = _device.logical().createPipelineLayout({
        {}, // flags
        setLayouts.size(),
        setLayouts.data(),
        1, // pushConstantRangeCount
        &pcRange
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
        &depthStencilState,
        &colorBlendState,
        nullptr, // dynamic
        _vkGraphicsPipelineLayout,
        _vkRenderPass,
        0 // subpass
    });

    _device.logical().destroyShaderModule(vertShaderModule);
    _device.logical().destroyShaderModule(fragShaderModule);
}

void App::createCommandBuffers(const SwapchainConfig& swapConfig)
{
    _vkCommandBuffers = _device.logical().allocateCommandBuffers({
        _device.commandPool(),
        vk::CommandBufferLevel::ePrimary,
        swapConfig.imageCount // commandBufferCount
    });
}

void App::drawFrame()
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t nextFrame = _swapchain.nextFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&]{
        auto nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[nextFrame]);
        while (!nextImage.has_value()) {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(_imageAvailableSemaphores[nextFrame]);
        }

        return nextImage.value();
    }();

    updateUniformBuffers(nextImage);

    recordCommandBuffer(nextImage);

    // Submit queue
    const std::array<vk::Semaphore, 1> waitSemaphores = {{
        _imageAvailableSemaphores[nextFrame]
    }};
    const std::array<vk::PipelineStageFlags, 1> waitStages = {{
        vk::PipelineStageFlagBits::eColorAttachmentOutput
    }};
    const std::array<vk::Semaphore, 1> signalSemaphores = {{
        _renderFinishedSemaphores[nextFrame]
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

    for (const auto& instance : _world.currentScene().modelInstances)
        instance.updateBuffer(&_device, nextImage);
}

void App::recordCommandBuffer(const uint32_t nextImage)
{
    const vk::CommandBuffer buffer = _vkCommandBuffers[nextImage];
    buffer.reset({});

    buffer.begin({
        vk::CommandBufferUsageFlagBits::eSimultaneousUse
    });

    const std::array<vk::ClearValue, 2> clearColors = {{
        {std::array<float, 4>{0.f, 0.f, 0.f, 0.f}}, // swap color
        {std::array<float, 4>{1.f, 0.f, 0.f, 0.f}} // depth
    }};
    buffer.beginRenderPass(
        {
            _vkRenderPass,
            _swapchain.fbo(nextImage),
            vk::Rect2D{
                {0, 0}, // offset
                _swapchain.extent()
            },
            clearColors.size(),
            clearColors.data()
        },
        vk::SubpassContents::eInline
    );

    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _vkGraphicsPipeline);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        _vkGraphicsPipelineLayout,
        0, // firstSet
        1, // descriptorSetCount
        &_cam.descriptorSet(nextImage),
        0, // dynamicOffsetCount
        nullptr // pDynamicOffsets
    );

    // Draw scene
    for (const auto& instance : _world.currentScene().modelInstances) {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            _vkGraphicsPipelineLayout,
            1, // firstSet
            1, // descriptorSetCount
            &instance.descriptorSets[nextImage],
            0, // dynamicOffsetCount
            nullptr // pDynamicOffsets
        );
        for (const auto& mesh : instance.model->_meshes) {
            buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                _vkGraphicsPipelineLayout,
                2, // firstSet
                1, // descriptorSetCount
                &mesh.material()._descriptorSet,
                0, // dynamicOffsetCount
                nullptr // pDynamicOffsets
            );
            Material::PCBlock pcBlock{
                mesh.material()._baseColorFactor,
                mesh.material()._metallicFactor,
                mesh.material()._roughnessFactor,
                mesh.material().alphaModeFloat(),
                mesh.material()._alphaCutoff,
                mesh.material()._texCoordSets.baseColor,
                mesh.material()._texCoordSets.metallicRoughness,
                mesh.material()._texCoordSets.normal
            };
            buffer.pushConstants(
                _vkGraphicsPipelineLayout,
                vk::ShaderStageFlagBits::eFragment,
                0, // offset
                sizeof(Material::PCBlock),
                &pcBlock
            );
            mesh.draw(buffer);
        }
    }

    buffer.endRenderPass();
    buffer.end();
}
