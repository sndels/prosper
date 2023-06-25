#include "VkUtils.hpp"

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
