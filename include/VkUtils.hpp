#ifndef PROSPER_VKUTILS_HPP
#define PROSPER_VKUTILS_HPP

#include <vulkan/vulkan.hpp>

void checkSuccess(vk::Result result, const char *source);

template <typename T, typename V> bool containsFlag(T mask, V flag)
{
    return (mask & flag) == flag;
}

template <typename T, typename V>
void assertContainsFlag(T mask, V flag, const char *errMsg)
{
    if (!containsFlag(mask, flag))
        throw std::runtime_error(errMsg);
}

constexpr vk::ImageAspectFlags aspectMask(vk::Format format)
{
    switch (format)
    {
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return vk::ImageAspectFlagBits::eDepth;
    case vk::Format::eS8Uint:
        return vk::ImageAspectFlagBits::eStencil;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return vk::ImageAspectFlagBits::eDepth |
               vk::ImageAspectFlagBits::eStencil;
    default:
        return vk::ImageAspectFlagBits::eColor;
    }
}

#endif // PROSPER_VKUTILS_HPP
