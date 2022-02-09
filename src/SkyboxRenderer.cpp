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
    fprintf(stderr, "Creating SkyboxRenderer\n");

    if (!compileShaders())
        throw std::runtime_error("SkyboxRenderer shader compilation failed");

    recreateSwapchainRelated(swapConfig, worldDSLayouts);
}

SkyboxRenderer::~SkyboxRenderer()
{
    if (_device)
    {
        destroySwapchainRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void SkyboxRenderer::recompileShaders(
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders())
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(swapConfig, worldDSLayouts);
    }
}

void SkyboxRenderer::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createAttachments();
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

    const std::array<vk::ImageMemoryBarrier2KHR, 2> barriers{
        _resources->images.sceneColor.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput,
            .accessMask = vk::AccessFlagBits2KHR::eColorAttachmentWrite,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        }),
        _resources->images.sceneDepth.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests,
            .accessMask = vk::AccessFlagBits2KHR::eDepthStencilAttachmentRead,
            .layout = vk::ImageLayout::eDepthAttachmentOptimal,
        }),
    };

    buffer.pipelineBarrier2KHR(vk::DependencyInfoKHR{
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });

    buffer.beginDebugUtilsLabelEXT(
        vk::DebugUtilsLabelEXT{.pLabelName = "Skybox"});

    buffer.beginRenderingKHR(vk::RenderingInfoKHR{
        .renderArea = renderArea,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &_colorAttachment,
        .pDepthAttachment = &_depthAttachment,
    });

    // Skybox doesn't need to be drawn under opaque geometry but should be
    // before transparents
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, _pipelineLayout,
        0, // firstSet
        1, &world._skyboxDSs[nextImage], 0, nullptr);

    world.drawSkybox(buffer);

    buffer.endRenderingKHR();

    buffer.endDebugUtilsLabelEXT(); // Skybox

    buffer.end();

    return buffer;
}

bool SkyboxRenderer::compileShaders()
{
    const auto vertSM =
        _device->compileShaderModule("shader/skybox.vert", "skyboxVS");
    const auto fragSM =
        _device->compileShaderModule("shader/skybox.frag", "skyboxPS");

    if (vertSM && fragSM)
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertSM,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragSM,
                .pName = "main",
            }};

        return true;
    }

    if (vertSM)
        _device->logical().destroy(*vertSM);
    if (fragSM)
        _device->logical().destroy(*fragSM);

    return false;
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

        destroyGraphicsPipelines();

        _colorAttachment = vk::RenderingAttachmentInfoKHR{};
        _depthAttachment = vk::RenderingAttachmentInfoKHR{};
    }
}

void SkyboxRenderer::destroyGraphicsPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void SkyboxRenderer::createAttachments()
{
    _colorAttachment = vk::RenderingAttachmentInfoKHR{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    _depthAttachment = vk::RenderingAttachmentInfoKHR{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

void SkyboxRenderer::createGraphicsPipelines(
    const SwapchainConfig &swapConfig, const World::DSLayouts &worldDSLayouts)
{
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

    const vk::PipelineRenderingCreateInfoKHR renderingCreateInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &_resources->images.sceneColor.format,
        .depthAttachmentFormat = _resources->images.sceneDepth.format,
    };

    const vk::GraphicsPipelineCreateInfo createInfo{
        .pNext = &renderingCreateInfo,
        .stageCount = static_cast<uint32_t>(_shaderStages.size()),
        .pStages = _shaderStages.data(),
        .pVertexInputState = &vertInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizerState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &colorBlendState,
        .layout = _pipelineLayout,
    };

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create skybox pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "SkyboxRenderer",
            });
    }
}

void SkyboxRenderer::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}
