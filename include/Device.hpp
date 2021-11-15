#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include "vulkan.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

#include <optional>
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

struct Buffer
{
    vk::Buffer handle;
    VmaAllocation allocation;
};

struct Image
{
    vk::Image handle;
    vk::ImageView view;
    vk::Format format;
    vk::Extent2D extent;
    VmaAllocation allocation = nullptr;
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

    void map(const VmaAllocation allocation, void **data) const;
    void unmap(const VmaAllocation allocation) const;

    Buffer createBuffer(
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        const VmaMemoryUsage vmaUsage) const;
    void destroy(const Buffer &buffer) const;

    Image createImage(
        const vk::Extent2D extent, const vk::Format format,
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
