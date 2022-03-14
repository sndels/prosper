#ifndef PROSPER_RESOURCES_HPP
#define PROSPER_RESOURCES_HPP

#include "vulkan.hpp"
#include <vk_mem_alloc.h>

struct BufferState
{
    vk::PipelineStageFlags2 stageMask{vk::PipelineStageFlagBits2::eTopOfPipe};
    vk::AccessFlags2 accessMask{vk::AccessFlagBits2::eNone};
};

struct Buffer
{
    vk::Buffer handle;
    VmaAllocation allocation{nullptr};
};

struct TexelBuffer
{
    vk::Buffer handle;
    vk::BufferView view;
    vk::Format format{vk::Format::eUndefined};
    vk::DeviceSize size{0};
    BufferState state;
    VmaAllocation allocation{nullptr};

    vk::BufferMemoryBarrier2 transitionBarrier(const BufferState &newState);
    void transition(vk::CommandBuffer buffer, const BufferState &newState);
};

struct ImageState
{
    vk::PipelineStageFlags2 stageMask{vk::PipelineStageFlagBits2::eTopOfPipe};
    vk::AccessFlags2 accessMask;
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
};

struct Image
{
    vk::Image handle;
    vk::ImageView view;
    vk::Format format{vk::Format::eUndefined};
    vk::Extent3D extent;
    vk::ImageSubresourceRange subresourceRange;
    ImageState state;
    VmaAllocation allocation{nullptr};

    vk::ImageMemoryBarrier2 transitionBarrier(const ImageState &newState);
    void transition(vk::CommandBuffer buffer, const ImageState &newState);
};

#endif // PROSPER_RESOURCES_HPP
