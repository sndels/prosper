#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include "vulkan.hpp"

#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>
#include <vk_mem_alloc.h>

#include <filesystem>
#include <optional>
#include <unordered_set>
#include <vector>

struct QueueFamilies
{
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const
    {
        return computeFamily.has_value() && graphicsFamily.has_value() &&
               presentFamily.has_value();
    }
};

struct BufferState
{
    vk::PipelineStageFlags2KHR stageMask{
        vk::PipelineStageFlagBits2KHR::eTopOfPipe};
    vk::AccessFlags2KHR accessMask{vk::AccessFlagBits2KHR::eNone};
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

    vk::BufferMemoryBarrier2KHR transitionBarrier(const BufferState &newState);
    void transition(
        const vk::CommandBuffer buffer, const BufferState &newState);
};

struct ImageState
{
    vk::PipelineStageFlags2KHR stageMask{
        vk::PipelineStageFlagBits2KHR::eTopOfPipe};
    vk::AccessFlags2KHR accessMask;
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

    vk::ImageMemoryBarrier2KHR transitionBarrier(const ImageState &newState);
    void transition(const vk::CommandBuffer buffer, const ImageState &newState);
};

class FileIncluder : public shaderc::CompileOptions::IncluderInterface
{
  public:
    FileIncluder();

    virtual shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type type,
        const char *requesting_source, size_t include_depth);

    virtual void ReleaseInclude(shaderc_include_result *data);

  private:
    std::filesystem::path _includePath;
};

class Device
{
  public:
    Device(GLFWwindow *window);
    ~Device();

    Device(const Device &other) = delete;
    Device &operator=(const Device &other) = delete;

    vk::Instance instance() const;
    vk::PhysicalDevice physical() const;
    vk::Device logical() const;
    vk::SurfaceKHR surface() const;
    vk::CommandPool computePool() const;
    vk::CommandPool graphicsPool() const;
    vk::Queue computeQueue() const;
    vk::Queue graphicsQueue() const;
    vk::Queue presentQueue() const;
    const QueueFamilies &queueFamilies() const;

    std::optional<vk::ShaderModule> compileShaderModule(
        const std::string &relPath, const std::string &debugName) const;
    std::optional<vk::ShaderModule> compileShaderModule(
        const std::string &source, const std::string &path,
        const std::string &debugName) const;

    void map(const VmaAllocation allocation, void **data) const;
    void unmap(const VmaAllocation allocation) const;

    Buffer createBuffer(
        const std::string &debugName, const vk::DeviceSize size,
        const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        const VmaMemoryUsage vmaUsage) const;
    void destroy(const Buffer &buffer) const;

    TexelBuffer createTexelBuffer(
        const std::string &debugName, const vk::Format format,
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties, const bool supportAtomics,
        const VmaMemoryUsage vmaUsage) const;
    void destroy(const TexelBuffer &buffer) const;

    Image createImage(
        const std::string &debugName, const vk::ImageType imageType,
        const vk::Extent3D extent, const vk::Format format,
        const vk::ImageSubresourceRange &range,
        const vk::ImageViewType viewType, const vk::ImageTiling tiling,
        const vk::ImageCreateFlags flags, const vk::ImageUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        const VmaMemoryUsage vmaUsage) const;
    void destroy(const Image &image) const;

    vk::CommandBuffer beginGraphicsCommands() const;
    void endGraphicsCommands(const vk::CommandBuffer buffer) const;

  private:
    bool isDeviceSuitable(const vk::PhysicalDevice device) const;

    void createInstance();
    void createDebugMessenger();
    void createSurface(GLFWwindow *window);
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    void createCommandPools();

    vk::Instance _instance;
    vk::PhysicalDevice _physical;
    vk::Device _logical;

    VmaAllocator _allocator = nullptr;

    shaderc::CompileOptions _compilerOptions;
    shaderc::Compiler _compiler;

    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies = {std::nullopt, std::nullopt, std::nullopt};
    vk::Queue _computeQueue;
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::CommandPool _computePool;
    vk::CommandPool _graphicsPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
