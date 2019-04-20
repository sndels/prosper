#include "Renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <fstream>

#include "Constants.hpp"

using namespace glm;

namespace {
    static std::vector<char> readFile(const std::string& filename)
    {
        // Open from end to find size from initial position
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error(std::string{"Failed to open file '"} + filename + "'");

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

Renderer::~Renderer()
{
    if (_device) {
        for (auto& semaphore : _renderFinishedSemaphores)
            _device->logical().destroy(semaphore);
        for (auto& semaphore : _imageAvailableSemaphores)
            _device->logical().destroy(semaphore);
    }
    destroySwapchainRelated();
}

void Renderer::init(Device* device)
{
    _device = device;
    // Semaphores correspond to logical frames instead of swapchain images
    createSemaphores(MAX_FRAMES_IN_FLIGHT);
}

void Renderer::createSwapchainRelated(const SwapchainConfig& swapConfig, const vk::DescriptorSetLayout camDSLayout, const World::DSLayouts& worldDSLayouts)
{
    createRenderPass(swapConfig);
    createGraphicsPipelines(swapConfig, camDSLayout, worldDSLayouts);
    // Each command buffer binds to specific swapchain image
    createCommandBuffers(swapConfig);
}

void Renderer::destroySwapchainRelated()
{
    if (_device) {
        _device->logical().freeCommandBuffers(
            _device->commandPool(),
            _commandBuffers.size(),
            _commandBuffers.data()
        );

        _device->logical().destroy(_pipelines.pbr);
        _device->logical().destroy(_pipelines.pbrAlphaBlend);
        _device->logical().destroy(_pipelines.skybox);
        _device->logical().destroy(_pipelineLayouts.pbr);
        _device->logical().destroy(_pipelineLayouts.skybox);
        _device->logical().destroy(_renderpass);
    }
}

vk::Semaphore Renderer::imageAvailable(const uint32_t frame) const
{
    return _imageAvailableSemaphores[frame];
}

vk::RenderPass Renderer::outputRenderpass() const
{
    return _renderpass;
}

std::array<vk::Semaphore, 1> Renderer::drawFrame(const World& world, const Camera& cam, const Swapchain& swapchain, const uint32_t nextImage) const
{
    updateUniformBuffers(world, cam, nextImage);

    recordCommandBuffer(world, cam, swapchain, nextImage);

    // Submit queue
    const size_t nextFrame = swapchain.nextFrame();

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
        &_commandBuffers[nextImage],
        signalSemaphores.size(),
        signalSemaphores.data()
    };
    _device->graphicsQueue().submit(1, &submitInfo, swapchain.currentFence());

    return signalSemaphores;
}

void Renderer::createRenderPass(const SwapchainConfig& swapConfig)
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

    _renderpass = _device->logical().createRenderPass({
        {}, // flags
        attachments.size(),
        attachments.data(),
        1, // subpassCount
        &subpass,
        1, // dependencyCount
        &dependency
    });
}

void Renderer::createGraphicsPipelines(const SwapchainConfig& swapConfig, const vk::DescriptorSetLayout camDSLayout, const World::DSLayouts& worldDSLayouts)
{
    {
        const auto vertSPV = readFile(binPath("shader/shader.vert.spv"));
        const auto fragSPV = readFile(binPath("shader/shader.frag.spv"));
        const vk::ShaderModule vertSM = createShaderModule(_device->logical(), vertSPV);
        const vk::ShaderModule fragSM = createShaderModule(_device->logical(), fragSPV);
        const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {{
            {
                {}, // flags
                vk::ShaderStageFlagBits::eVertex,
                vertSM,
                "main"
            },
            {
                {}, // flags
                vk::ShaderStageFlagBits::eFragment,
                fragSM,
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

        // Alpha blend is created with a modified version
        vk::PipelineRasterizationStateCreateInfo rasterizerState{
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

        // Alpha blend pipeline is created with a modified version
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
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
            camDSLayout,
            worldDSLayouts.modelInstance,
            worldDSLayouts.material
        }};
        const vk::PushConstantRange pcRange{
            vk::ShaderStageFlagBits::eFragment,
            0, // offset
            sizeof(Material::PCBlock)
        };
        _pipelineLayouts.pbr = _device->logical().createPipelineLayout({
            {}, // flags
            setLayouts.size(),
            setLayouts.data(),
            1, // pushConstantRangeCount
            &pcRange
        });

        const vk::GraphicsPipelineCreateInfo createInfo{
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
            _pipelineLayouts.pbr,
            _renderpass,
            0 // subpass
        };
        _pipelines.pbr = _device->logical().createGraphicsPipeline({}, createInfo);

        rasterizerState.cullMode = vk::CullModeFlagBits::eNone;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
        colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;

        _pipelines.pbrAlphaBlend = _device->logical().createGraphicsPipeline({}, createInfo);

        _device->logical().destroyShaderModule(vertSM);
        _device->logical().destroyShaderModule(fragSM);
    }

    {
        const auto vertSPV = readFile(binPath("shader/skybox.vert.spv"));
        const auto fragSPV = readFile(binPath("shader/skybox.frag.spv"));
        const vk::ShaderModule vertSM = createShaderModule(_device->logical(), vertSPV);
        const vk::ShaderModule fragSM = createShaderModule(_device->logical(), fragSPV);
        const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {{
            {
                {}, // flags
                vk::ShaderStageFlagBits::eVertex,
                vertSM,
                "main"
            },
            {
                {}, // flags
                vk::ShaderStageFlagBits::eFragment,
                fragSM,
                "main"
            }
        }};

        const vk::VertexInputBindingDescription vertexBindingDescription{
            0,
            sizeof(vec3), // Only position
            vk::VertexInputRate::eVertex
        };
        const vk::VertexInputAttributeDescription vertexAttributeDescription{
            0, // location
            0, // binding
            vk::Format::eR32G32B32Sfloat,
            0
        };
        const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
            {}, // flags
            1, // vertexBindingDescriptionCount
            &vertexBindingDescription,
            1, //vertexAttributeDescriptionCount
            &vertexAttributeDescription
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
            vk::CullModeFlagBits::eNone, // Draw the skybox from inside
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
            vk::CompareOp::eLessOrEqual // Draw skybox at maximum depth
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

        _pipelineLayouts.skybox = _device->logical().createPipelineLayout({
            {}, // flags
            1, // layoutCount
            &worldDSLayouts.skybox
        });

        const vk::GraphicsPipelineCreateInfo createInfo{
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
            _pipelineLayouts.skybox,
            _renderpass,
            0 // subpass
        };
        _pipelines.skybox = _device->logical().createGraphicsPipeline({}, createInfo);

        _device->logical().destroyShaderModule(vertSM);
        _device->logical().destroyShaderModule(fragSM);
    }
}

void Renderer::createCommandBuffers(const SwapchainConfig& swapConfig)
{
    _commandBuffers = _device->logical().allocateCommandBuffers({
        _device->commandPool(),
        vk::CommandBufferLevel::ePrimary,
        swapConfig.imageCount // commandBufferCount
    });
}

void Renderer::createSemaphores(const uint32_t concurrentFrameCount)
{
    for (size_t i = 0; i < concurrentFrameCount; ++i) {
        _imageAvailableSemaphores.push_back(_device->logical().createSemaphore({}));
        _renderFinishedSemaphores.push_back(_device->logical().createSemaphore({}));
    }
}

void Renderer::updateUniformBuffers(const World& world, const Camera& cam, const uint32_t nextImage) const
{
    cam.updateBuffer(nextImage);

    const mat4 worldToClip = cam.cameraToClip() * mat4(mat3(cam.worldToCamera()));
    void* data;
    _device->map(world._skyboxUniformBuffers[nextImage].allocation, &data);
    memcpy(data, &worldToClip, sizeof(mat4));
    _device->unmap(world._skyboxUniformBuffers[nextImage].allocation);

    for (const auto& instance : world.currentScene().modelInstances)
        instance.updateBuffer(_device, nextImage);
}

void Renderer::recordCommandBuffer(const World& world, const Camera& cam, const Swapchain& swapchain, const uint32_t nextImage) const
{
    const auto buffer = _commandBuffers[nextImage];
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
            _renderpass,
            swapchain.fbo(nextImage),
            vk::Rect2D{
                {0, 0}, // offset
                swapchain.extent()
            },
            clearColors.size(),
            clearColors.data()
        },
        vk::SubpassContents::eInline
    );

    // Draw opaque and alpha masked geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.pbr);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        _pipelineLayouts.pbr,
        0, // firstSet
        1, // descriptorSetCount
        &cam.descriptorSet(nextImage),
        0, // dynamicOffsetCount
        nullptr // pDynamicOffsets
    );

    recordModelInstances(
        buffer,
        nextImage,
        world.currentScene().modelInstances,
        [](const Mesh& mesh){
            return mesh.material()._alphaMode == Material::AlphaMode::Blend;
        }
    );

    // Skybox doesn't need to be drawn under opaque geometry but should be before transparents
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.skybox);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        _pipelineLayouts.skybox,
        0, // firstSet
        1, // descriptorSetCount
        &world._skyboxDSs[nextImage],
        0, // dynamicOffsetCount
        nullptr // pDynamicOffsets
    );

    world.drawSkybox(buffer);

    // Draw transparent geometry
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipelines.pbrAlphaBlend);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        _pipelineLayouts.pbr,
        0, // firstSet
        1, // descriptorSetCount
        &cam.descriptorSet(nextImage),
        0, // dynamicOffsetCount
        nullptr // pDynamicOffsets
    );

    // TODO: Sort back to front
    recordModelInstances(
        buffer,
        nextImage,
        world.currentScene().modelInstances,
        [](const Mesh& mesh){
            return mesh.material()._alphaMode != Material::AlphaMode::Blend;
        }
    );

    buffer.endRenderPass();
    buffer.end();
}

void Renderer::recordModelInstances(const vk::CommandBuffer buffer, const uint32_t nextImage, const std::vector<Scene::ModelInstance>& instances, const std::function<bool(const Mesh&)>& cullMesh) const
{
    for (const auto& instance : instances) {
        buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            _pipelineLayouts.pbr,
            1, // firstSet
            1, // descriptorSetCount
            &instance.descriptorSets[nextImage],
            0, // dynamicOffsetCount
            nullptr // pDynamicOffsets
        );
        for (const auto& mesh : instance.model->_meshes) {
            if (cullMesh(mesh))
                continue;
            buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                _pipelineLayouts.pbr,
                2, // firstSet
                1, // descriptorSetCount
                &mesh.material()._descriptorSet,
                0, // dynamicOffsetCount
                nullptr // pDynamicOffsets
            );
            const auto pcBlock = mesh.material().pcBlock();
            buffer.pushConstants(
                _pipelineLayouts.pbr,
                vk::ShaderStageFlagBits::eFragment,
                0, // offset
                sizeof(Material::PCBlock),
                &pcBlock
            );
            mesh.draw(buffer);
        }
    }
}
