#include "VkUtils.hpp"

#include <wheels/containers/static_array.hpp>

#include "../utils/Utils.hpp"

using namespace wheels;

void setViewportScissor(vk::CommandBuffer cb, const vk::Rect2D &area)
{
    const vk::Viewport viewport{
        .x = static_cast<float>(area.offset.x),
        .y = static_cast<float>(area.offset.y),
        .width = static_cast<float>(area.extent.width),
        .height = static_cast<float>(area.extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    cb.setViewport(0, 1, &viewport);
    cb.setScissor(0, 1, &area);
}

vk::Pipeline createComputePipeline(
    vk::Device device, const vk::ComputePipelineCreateInfo &createInfo,
    const char *debugName)
{
    const vk::ResultValue<vk::Pipeline> pipeline =
        device.createComputePipeline(vk::PipelineCache{}, createInfo);
    if (pipeline.result != vk::Result::eSuccess)
        throw std::runtime_error(
            std::string{"Failed to create pipeline '"} + debugName + "'");

    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::ePipeline,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkPipeline>(pipeline.value)),
        .pObjectName = debugName,
    });

    return pipeline.value;
}

vk::Pipeline createGraphicsPipeline(
    vk::Device device, const GraphicsPipelineInfo &info)
{
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = info.topology,
    };

    // Dynamic state
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizerState{
        .lineWidth = 1.0,
    };
    if (info.topology == vk::PrimitiveTopology::eTriangleList)
    {
        rasterizerState.polygonMode = vk::PolygonMode::eFill;
        rasterizerState.cullMode = info.cullMode;
        rasterizerState.frontFace = vk::FrontFace::eCounterClockwise;
    }
    else
        WHEELS_ASSERT(
            info.topology == vk::PrimitiveTopology::eLineList &&
            "Expected triangle list or line list");

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = info.writeDepth ? VK_TRUE : VK_FALSE,
        .depthCompareOp = info.depthCompareOp,
    };

    const vk::PipelineColorBlendStateCreateInfo opaqueColorBlendState{
        .attachmentCount =
            asserted_cast<uint32_t>(info.colorBlendAttachments.size()),
        .pAttachments = info.colorBlendAttachments.data(),
    };

    const StaticArray dynamicStates{
        {vk::DynamicState::eViewport, vk::DynamicState::eScissor}};

    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = asserted_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    const vk::StructureChain<
        vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo>
        pipelineChain{
            vk::GraphicsPipelineCreateInfo{
                .stageCount = asserted_cast<uint32_t>(info.shaderStages.size()),
                .pStages = info.shaderStages.data(),
                .pVertexInputState = &info.vertInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizerState,
                .pMultisampleState = &multisampleState,
                .pDepthStencilState = &depthStencilState,
                .pColorBlendState = &opaqueColorBlendState,
                .pDynamicState = &dynamicState,
                .layout = info.layout,
            },
            info.renderingInfo};

    const vk::ResultValue<vk::Pipeline> pipeline =
        device.createGraphicsPipeline(
            vk::PipelineCache{},
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
    if (pipeline.result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to create pbr pipeline");

    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
        .objectType = vk::ObjectType::ePipeline,
        .objectHandle =
            reinterpret_cast<uint64_t>(static_cast<VkPipeline>(pipeline.value)),
        .pObjectName = info.debugName,
    });

    return pipeline.value;
}
