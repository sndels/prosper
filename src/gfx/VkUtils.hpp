#ifndef PROSPER_GFX_VKUTILS_HPP
#define PROSPER_GFX_VKUTILS_HPP

#include <vulkan/vulkan.hpp>

constexpr void checkSuccess(vk::Result result, const char *source)
{
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            std::string(source) + " failed: " + vk::to_string(result) + "!");
    }
}

template <typename T, typename V> constexpr bool containsFlag(T mask, V flag)
{
    return (mask & flag) == flag;
}

template <typename T, typename V>
constexpr void assertContainsFlag(T mask, V flag, const char *errMsg)
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

void setViewportScissor(vk::CommandBuffer cb, const vk::Rect2D &area);

constexpr vk::PipelineColorBlendAttachmentState opaqueColorBlendAttachment()
{
    return vk::PipelineColorBlendAttachmentState{
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
}

constexpr vk::PipelineColorBlendAttachmentState
transparentColorBlendAttachment()
{
    return vk::PipelineColorBlendAttachmentState{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
}

#endif // PROSPER_GFX_VKUTILS_HPP
