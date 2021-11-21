#ifndef PROSPER_VKUTILS_HPP
#define PROSPER_VKUTILS_HPP

#include "Device.hpp"

void transitionImageLayout(
    const vk::CommandBuffer &commandBuffer, const vk::Image &image,
    const vk::ImageSubresourceRange &subresourceRange,
    const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
    const vk::AccessFlags2KHR srcAccessMask,
    const vk::AccessFlags2KHR dstAccessMask,
    const vk::PipelineStageFlags2KHR srcStageMask,
    const vk::PipelineStageFlags2KHR dstStageMask);

void checkSuccess(vk::Result result, const char *source);

#endif // PROSPER_VKUTILS_HPP
