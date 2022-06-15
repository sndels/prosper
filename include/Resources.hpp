#ifndef PROSPER_RESOURCES_HPP
#define PROSPER_RESOURCES_HPP

#include "vulkan.hpp"
#include <vk_mem_alloc.h>

// TODO: Get rid of these, could be inferred from MemoryPropertyFlags
enum class MemoryAccess
{
    Device,
    HostSequentialWrite,
    HostRandomWrite,
};

struct BufferState
{
    vk::PipelineStageFlags2 stageMask{vk::PipelineStageFlagBits2::eTopOfPipe};
    vk::AccessFlags2 accessMask{vk::AccessFlagBits2::eNone};
};

struct BufferCreateInfo
{
    vk::DeviceSize byteSize{0};
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags properties;
    MemoryAccess access{MemoryAccess::Device};
    const void *initialData{nullptr};
    bool createMapped{false};
    std::string debugName;
};

struct Buffer
{
    vk::Buffer handle;
    void *mapped{nullptr};
    VmaAllocation allocation{nullptr};
};

struct TexelBufferCreateInfo
{
    BufferCreateInfo bufferInfo;
    vk::Format format;
    bool supportAtomics{false};
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

struct ImageCreateInfo
{
    vk::ImageType imageType{vk::ImageType::e2D};
    vk::Format format{vk::Format::eUndefined};
    uint32_t width{1};
    uint32_t height{1};
    uint32_t depth{1};
    uint32_t mipCount{1};
    uint32_t layerCount{1};
    vk::ImageCreateFlags createFlags;
    vk::ImageUsageFlags usageFlags;
    vk::MemoryPropertyFlags properties{
        vk::MemoryPropertyFlagBits::eDeviceLocal};

    std::string debugName;
};

struct Image
{
    vk::Image handle;
    vk::ImageView view;
    vk::Format format{vk::Format::eUndefined};
    // Keep extent and subresource range in full to avoid having to refill them
    // on every use
    vk::Extent3D extent;
    vk::ImageSubresourceRange subresourceRange;
    ImageState state;
    VmaAllocation allocation{nullptr};

    vk::ImageMemoryBarrier2 transitionBarrier(const ImageState &newState);
    void transition(vk::CommandBuffer buffer, const ImageState &newState);
};

struct AccelerationStructure
{
    vk::AccelerationStructureKHR handle;
    Buffer buffer;
};

#endif // PROSPER_RESOURCES_HPP
