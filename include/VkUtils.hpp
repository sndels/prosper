#ifndef PROSPER_VKUTILS_HPP
#define PROSPER_VKUTILS_HPP

#include "Device.hpp"

void transitionImageLayout(const vk::Image& image, const vk::ImageSubresourceRange& subresourceRange, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, const vk::CommandBuffer& commandBuffer);

#endif // PROSPER_VKUTILS_HPP
