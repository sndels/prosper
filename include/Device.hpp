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

    [[nodiscard]] bool isComplete() const
    {
        return computeFamily.has_value() && graphicsFamily.has_value() &&
               presentFamily.has_value();
    }
};

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

struct DeviceProperties
{
    vk::PhysicalDeviceProperties device;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtPipeline;
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure;
};

class FileIncluder : public shaderc::CompileOptions::IncluderInterface
{
  public:
    FileIncluder();

    shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type type,
        const char *requesting_source, size_t include_depth) override;

    void ReleaseInclude(shaderc_include_result *data) override;

  private:
    std::filesystem::path _includePath;

    uint64_t _includeContentID{0};
    std::unordered_map<
        uint64_t,
        std::tuple<std::shared_ptr<shaderc_include_result>, std::string>>
        _includeContent;
};

class Device
{
  public:
    Device(GLFWwindow *window);
    ~Device();

    Device(const Device &other) = delete;
    Device(Device &&other) = delete;
    Device &operator=(const Device &other) = delete;
    Device &operator=(Device &&other) = delete;

    [[nodiscard]] vk::Instance instance() const;
    [[nodiscard]] vk::PhysicalDevice physical() const;
    [[nodiscard]] vk::Device logical() const;
    [[nodiscard]] vk::SurfaceKHR surface() const;
    [[nodiscard]] vk::CommandPool computePool() const;
    [[nodiscard]] vk::CommandPool graphicsPool() const;
    [[nodiscard]] vk::Queue computeQueue() const;
    [[nodiscard]] vk::Queue graphicsQueue() const;
    [[nodiscard]] vk::Queue presentQueue() const;
    [[nodiscard]] const QueueFamilies &queueFamilies() const;
    [[nodiscard]] const DeviceProperties &properties() const;

    [[nodiscard]] std::optional<vk::ShaderModule> compileShaderModule(
        const std::string &relPath, const std::string &debugName) const;
    [[nodiscard]] std::optional<vk::ShaderModule> compileShaderModule(
        const std::string &source, const std::string &path,
        const std::string &debugName) const;

    void map(VmaAllocation allocation, void **data) const;
    void unmap(VmaAllocation allocation) const;

    [[nodiscard]] Buffer createBuffer(
        const std::string &debugName, vk::DeviceSize size,
        vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
        VmaMemoryUsage vmaUsage) const;
    void destroy(const Buffer &buffer) const;

    [[nodiscard]] TexelBuffer createTexelBuffer(
        const std::string &debugName, vk::Format format, vk::DeviceSize size,
        vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
        bool supportAtomics, VmaMemoryUsage vmaUsage) const;
    void destroy(const TexelBuffer &buffer) const;

    [[nodiscard]] Image createImage(
        const std::string &debugName, vk::ImageType imageType,
        const vk::Extent3D &extent, vk::Format format,
        const vk::ImageSubresourceRange &range, vk::ImageViewType viewType,
        vk::ImageTiling tiling, vk::ImageCreateFlags flags,
        vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
        VmaMemoryUsage vmaUsage) const;
    void destroy(const Image &image) const;

    [[nodiscard]] vk::CommandBuffer beginGraphicsCommands() const;
    void endGraphicsCommands(vk::CommandBuffer buffer) const;

  private:
    [[nodiscard]] bool isDeviceSuitable(vk::PhysicalDevice device) const;

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
    DeviceProperties _properties;

    VmaAllocator _allocator{nullptr};

    shaderc::CompileOptions _compilerOptions;
    shaderc::Compiler _compiler;

    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies;
    vk::Queue _computeQueue;
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::CommandPool _computePool;
    vk::CommandPool _graphicsPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
