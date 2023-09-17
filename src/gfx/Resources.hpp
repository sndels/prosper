#ifndef PROSPER_GFX_RESOURCES_HPP
#define PROSPER_GFX_RESOURCES_HPP

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/span.hpp>

// TODO:
// Tighter transfer, shader access flags
enum class BufferState : uint32_t
{
    Unknown = 0,

    // Stages
    StageFragmentShader = 0x1,
    StageComputeShader = 0x2,
    // Covers copy, blit, resolve and clear
    StageTransfer = 0x4,

    // Access
    // Covers sampled and storage reads
    AccessShaderRead = 0x8,
    AccessShaderWrite = 0x10,
    // Covers copy, blit, resolve and clear
    AccessTransferRead = 0x20,
    // Covers copy, blit, resolve and clear
    AccessTransferWrite = 0x40,

    // Combined Masks
    FragmentShaderRead = StageFragmentShader | AccessShaderRead,
    ComputeShaderRead = StageComputeShader | AccessShaderRead,
    ComputeShaderWrite = StageComputeShader | AccessShaderWrite,
    ComputeShaderReadWrite = ComputeShaderRead | ComputeShaderWrite,
    TransferSrc = StageTransfer | AccessTransferRead,
    TransferDst = StageTransfer | AccessTransferWrite,
};

// TODO:
// Tighter transfer, shader access flags
enum class ImageState : uint32_t
{
    Unknown = 0,

    // Stages
    StageFragmentShader = 0x1,
    StageEarlyFragmentTests = 0x2,
    StageLateFragmentTests = 0x4,
    StageColorAttachmentOutput = 0x8,
    StageComputeShader = 0x10,
    StageRayTracingShader = 0x20,
    // Covers copy, blit, resolve and clear
    StageTransfer = 0x40,

    // Access
    // Covers sampled and storage reads
    AccessShaderRead = 0x80,
    AccessShaderWrite = 0x100,
    AccessColorAttachmentRead = 0x200,
    AccessColorAttachmentWrite = 0x400,
    AccessDepthAttachmentRead = 0x800,
    AccessDepthAttachmentWrite = 0x1000,
    // Covers copy, blit, resolve and clear
    AccessTransferRead = 0x2000,
    // Covers copy, blit, resolve and clear
    AccessTransferWrite = 0x4000,

    // Combined Masks
    ColorAttachmentWrite =
        StageColorAttachmentOutput | AccessColorAttachmentWrite,
    ColorAttachmentReadWrite = StageColorAttachmentOutput |
                               AccessColorAttachmentRead |
                               AccessColorAttachmentWrite,
    DepthAttachmentRead = StageEarlyFragmentTests | AccessDepthAttachmentRead,
    DepthAttachmentWrite = StageLateFragmentTests | AccessDepthAttachmentWrite,
    DepthAttachmentReadWrite = DepthAttachmentRead | DepthAttachmentWrite,
    FragmentShaderRead = StageFragmentShader | AccessShaderRead,
    ComputeShaderRead = StageComputeShader | AccessShaderRead,
    ComputeShaderWrite = StageComputeShader | AccessShaderWrite,
    ComputeShaderReadWrite = ComputeShaderRead | ComputeShaderWrite,
    RayTracingRead = StageRayTracingShader | AccessShaderRead,
    RayTracingWrite = StageRayTracingShader | AccessShaderWrite,
    RayTracingReadWrite = RayTracingRead | RayTracingWrite,
    TransferSrc = StageTransfer | AccessTransferRead,
    TransferDst = StageTransfer | AccessTransferWrite,
};

template <typename T>
    requires(wheels::SameAs<T, BufferState> || wheels::SameAs<T, ImageState>)
constexpr T operator&(T lhs, T rhs)
{
    return static_cast<T>(
        static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

template <typename T>
    requires(wheels::SameAs<T, BufferState> || wheels::SameAs<T, ImageState>)
constexpr T operator|(T lhs, T rhs)
{
    return static_cast<T>(
        static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

template <typename T>
    requires(wheels::SameAs<T, BufferState> || wheels::SameAs<T, ImageState>)
constexpr bool contains(T state, T subState)
{
    return (state & subState) == subState;
}

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
    BufferState state{BufferState::Unknown};
    VmaAllocation allocation{nullptr};

    [[nodiscard]] wheels::Optional<vk::BufferMemoryBarrier2> transitionBarrier(
        BufferState newState, bool force_barrier = false);
    void transition(vk::CommandBuffer cb, BufferState newState);
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
    BufferState state{BufferState::Unknown};
    VmaAllocation allocation{nullptr};

    [[nodiscard]] wheels::Optional<vk::BufferMemoryBarrier2> transitionBarrier(
        BufferState newState, bool force_barrier = false);
    void transition(vk::CommandBuffer cb, BufferState newState);
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
    ImageState state{ImageState::Unknown};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize rawByteSize{0};

    [[nodiscard]] wheels::Optional<vk::ImageMemoryBarrier2> transitionBarrier(
        ImageState newState, bool force_barrier = false);
    void transition(vk::CommandBuffer buffer, ImageState newState);
};

struct AccelerationStructure
{
    vk::AccelerationStructureKHR handle;
    Buffer buffer;
};

#endif // PROSPER_GFX_RESOURCES_HPP