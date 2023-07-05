#ifndef PROSPER_RESOURCES_HPP
#define PROSPER_RESOURCES_HPP

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <wheels/containers/span.hpp>

struct BufferState
{
    vk::PipelineStageFlags2 stageMask{vk::PipelineStageFlagBits2::eTopOfPipe};
    vk::AccessFlags2 accessMask{vk::AccessFlagBits2::eNone};
};

struct BufferDescription
{
    vk::DeviceSize byteSize{0};
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags properties;

    [[nodiscard]] bool matches(const BufferDescription &other) const
    {
        if (byteSize != other.byteSize)
            return false;
        if (usage != other.usage)
            return false;
        if (properties != other.properties)
            return false;
        return true;
    }
};

struct BufferCreateInfo
{
    BufferDescription desc;
    // TODO: When readback is needed, add enum for gpuonly, staging, readback to
    // select vma allocation mode accordingly
    const void *initialData{nullptr};
    bool createMapped{false};

    const char *debugName{nullptr};
};

struct Buffer
{
    vk::Buffer handle;
    vk::DeviceSize byteSize{0};
    void *mapped{nullptr};
    BufferState state;
    VmaAllocation allocation{nullptr};

    [[nodiscard]] vk::BufferMemoryBarrier2 transitionBarrier(
        const BufferState &newState);
    void transition(vk::CommandBuffer cb, const BufferState &newState);
};

struct TexelBufferDescription
{
    BufferDescription bufferDesc;
    vk::Format format{vk::Format::eUndefined};
    bool supportAtomics{false};

    [[nodiscard]] bool matches(const TexelBufferDescription &other) const
    {
        if (!bufferDesc.matches(other.bufferDesc))
            return false;
        if (format != other.format)
            return false;
        if (supportAtomics != other.supportAtomics)
            return false;
        return true;
    }
};

struct TexelBufferCreateInfo
{
    TexelBufferDescription desc;

    const char *debugName{nullptr};
};

struct TexelBuffer
{
    vk::Buffer handle;
    vk::BufferView view;
    vk::Format format{vk::Format::eUndefined};
    vk::DeviceSize size{0};
    BufferState state;
    VmaAllocation allocation{nullptr};

    [[nodiscard]] vk::BufferMemoryBarrier2 transitionBarrier(
        const BufferState &newState);
    void transition(vk::CommandBuffer cb, const BufferState &newState);
};

struct ImageState
{
    vk::PipelineStageFlags2 stageMask{vk::PipelineStageFlagBits2::eTopOfPipe};
    vk::AccessFlags2 accessMask;
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
};

struct ImageDescription
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

    [[nodiscard]] bool matches(const ImageDescription &other) const
    {
        if (imageType != other.imageType)
            return false;
        if (format != other.format)
            return false;
        if (width != other.width)
            return false;
        if (height != other.height)
            return false;
        if (depth != other.depth)
            return false;
        if (mipCount != other.mipCount)
            return false;
        if (layerCount != other.layerCount)
            return false;
        if (createFlags != other.createFlags)
            return false;
        if (usageFlags != other.usageFlags)
            return false;
        if (properties != other.properties)
            return false;
        return true;
    }
};

struct ImageCreateInfo
{
    ImageDescription desc;

    const char *debugName{nullptr};
};

struct Image
{
    vk::Image handle;
    vk::ImageView view;
    vk::ImageType imageType{vk::ImageType::e2D};
    vk::Format format{vk::Format::eUndefined};
    // Keep extent and subresource range in full to avoid having to refill them
    // on every use
    vk::Extent3D extent;
    vk::ImageSubresourceRange subresourceRange;
    ImageState state;
    VmaAllocation allocation{nullptr};
    vk::DeviceSize rawByteSize{0};

    [[nodiscard]] vk::ImageMemoryBarrier2 transitionBarrier(
        const ImageState &newState);
    void transition(vk::CommandBuffer buffer, const ImageState &newState);
};

struct AccelerationStructure
{
    vk::AccelerationStructureKHR handle;
    Buffer buffer;
};

#endif // PROSPER_RESOURCES_HPP
