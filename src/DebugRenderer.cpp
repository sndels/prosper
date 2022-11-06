#include "DebugRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "LightClustering.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

namespace
{

constexpr uint32_t sCameraBindingSet = 0;
constexpr uint32_t sGeometryBuffersBindingSet = 1;

} // namespace

DebugRenderer::DebugRenderer(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating DebugRenderer\n");

    if (!compileShaders())
        throw std::runtime_error("DebugRenderer shader compilation failed");

    recreate(swapConfig, camDSLayout);
}

DebugRenderer::~DebugRenderer()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void DebugRenderer::recompileShaders(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout)
{
    if (compileShaders())
    {
        destroyGraphicsPipeline();
        createGraphicsPipeline(swapConfig, camDSLayout);
    }
}

void DebugRenderer::recreate(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout)
{
    destroySwapchainRelated();

    createBuffers(swapConfig);
    createDescriptorSets(swapConfig.imageCount);
    createAttachments();
    createGraphicsPipeline(swapConfig, camDSLayout);
}

void DebugRenderer::record(
    vk::CommandBuffer cb, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage, Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const auto _s = profiler->createCpuGpuScope(cb, "Debug");

        const std::array imageBarriers{
            _resources->images.sceneColor.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
            _resources->images.sceneDepth.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .layout = vk::ImageLayout::eDepthAttachmentOptimal,
            }),
        };

        const auto &lines = _resources->buffers.debugLines[nextImage];
        // No need for barrier, mapped writes

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount =
                asserted_cast<uint32_t>(imageBarriers.size()),
            .pImageMemoryBarriers = imageBarriers.data(),
        });

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &_colorAttachment,
            .pDepthAttachment = &_depthAttachment,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        std::array<vk::DescriptorSet, 2> descriptorSets = {};
        descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextImage);
        descriptorSets[sGeometryBuffersBindingSet] =
            _linesDescriptorSets[nextImage];

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        cb.draw(lines.count * 2, 1, 0, 0);

        cb.endRendering();
    }
}

bool DebugRenderer::compileShaders()
{
    printf("Compiling DebugRenderer shaders\n");

    std::string vertDefines;
    vertDefines += defineStr("CAMERA_SET", sCameraBindingSet);
    vertDefines += defineStr("GEOMETRY_SET", sGeometryBuffersBindingSet);
    const auto vertSM =
        _device->compileShaderModule(Device::CompileShaderModuleArgs{
            .relPath = "shader/debug_lines.vert",
            .debugName = "debugLinesVS",
            .defines = vertDefines,
        });

    const auto fragSM =
        _device->compileShaderModule(Device::CompileShaderModuleArgs{
            .relPath = "shader/debug_color.frag",
            .debugName = "debugColorPS",
        });

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

void DebugRenderer::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipeline();

        _device->logical().destroy(_linesDSLayout);
        _linesDescriptorSets.clear();

        for (auto &ls : _resources->buffers.debugLines)
            _device->destroy(ls.buffer);
        _resources->buffers.debugLines.clear();
    }
}

void DebugRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DebugRenderer::createBuffers(const SwapchainConfig &swapConfig)
{
    for (auto i = 0u; i < swapConfig.imageCount; ++i)
        _resources->buffers.debugLines.push_back(DebugLines{
            .buffer = _device->createBuffer(BufferCreateInfo{
                .byteSize = DebugLines::sMaxLineCount * DebugLines::sLineBytes,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eHostCoherent |
                              vk::MemoryPropertyFlagBits::eHostVisible,
                .createMapped = true,
                .debugName = "DebugLines",
            }),
        });
}

void DebugRenderer::createDescriptorSets(const uint32_t swapImageCount)
{
    const vk::DescriptorSetLayoutBinding layoutBinding{
        .binding = 0, // binding
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1, // descriptorCount
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };
    _linesDSLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &layoutBinding,
        });

    const std::vector<vk::DescriptorSetLayout> layouts(
        swapImageCount, _linesDSLayout);
    _linesDescriptorSets =
        _resources->descriptorAllocator.allocate(std::span{layouts});

    for (size_t i = 0; i < _linesDescriptorSets.size(); ++i)
    {
        vk::DescriptorBufferInfo info{
            .buffer = _resources->buffers.debugLines[i].buffer.handle,
            .range = VK_WHOLE_SIZE,
        };

        const vk::WriteDescriptorSet descriptorWrite{
            .dstSet = _linesDescriptorSets[i],
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &info,
        };
        _device->logical().updateDescriptorSets(
            1, &descriptorWrite, 0, nullptr);
    }
}

void DebugRenderer::createAttachments()
{
    _colorAttachment = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneColor.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    _depthAttachment = vk::RenderingAttachmentInfo{
        .imageView = _resources->images.sceneDepth.view,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

void DebugRenderer::createGraphicsPipeline(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout)
{
    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eLineList,
    };

    // TODO: Dynamic viewport state?
    const vk::Viewport viewport{
        .x = 0.f,
        .y = 0.f,
        .width = static_cast<float>(swapConfig.extent.width),
        .height = static_cast<float>(swapConfig.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapConfig.extent,
    };
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    const vk::PipelineRasterizationStateCreateInfo rasterizerState{
        .lineWidth = 1.0,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess,
    };

    const vk::PipelineColorBlendAttachmentState opaqueColorBlendAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo opaqueColorBlendState{
        .attachmentCount = 1,
        .pAttachments = &opaqueColorBlendAttachment,
    };

    std::array<vk::DescriptorSetLayout, 2> setLayouts = {};
    setLayouts[sCameraBindingSet] = camDSLayout;
    setLayouts[sGeometryBuffersBindingSet] = _linesDSLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo>
        pipelineChain{
            vk::GraphicsPipelineCreateInfo{
                .stageCount = asserted_cast<uint32_t>(_shaderStages.size()),
                .pStages = _shaderStages.data(),
                .pVertexInputState = &vertInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizerState,
                .pMultisampleState = &multisampleState,
                .pDepthStencilState = &depthStencilState,
                .pColorBlendState = &opaqueColorBlendState,
                .layout = _pipelineLayout,
            },
            vk::PipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats =
                    &_resources->images.sceneColor.format,
                .depthAttachmentFormat = _resources->images.sceneDepth.format,
            }};

    {
        auto pipeline = _device->logical().createGraphicsPipeline(
            vk::PipelineCache{},
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "DebugRenderer::Lines",
            });
    }
}
