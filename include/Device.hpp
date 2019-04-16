#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include <vulkan/vulkan.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

#include <optional>
#include <vector>

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct Buffer {
    vk::Buffer handle;
    VmaAllocation allocation;
};

struct Image {
    vk::Image handle;
    VmaAllocation allocation;
};

class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device& other) = delete;
    Device& operator=(const Device& other) = delete;

    void init(GLFWwindow* window);

    vk::Instance instance() const;
    vk::PhysicalDevice physical() const;
    vk::Device logical() const;
    vk::SurfaceKHR surface() const;
    vk::CommandPool commandPool() const;
    vk::Queue graphicsQueue() const;
    vk::Queue presentQueue() const;
    const QueueFamilies& queueFamilies() const;

    void map(const VmaAllocation allocation, void** data) const;
    void unmap(const VmaAllocation allocation) const;

    Buffer createBuffer(const vk::DeviceSize size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties, const VmaMemoryUsage vmaUsage) const;
    void copyBuffer(const Buffer& src, const Buffer& dst, const vk::DeviceSize size) const;
    void copyBufferToImage(const Buffer& src, const Image& dst, const vk::Extent2D extent) const;
    void destroy(const Buffer& buffer);

    Image createImage(const vk::Extent2D extent, const uint32_t mipLevels, const uint32_t arrayLayers, const vk::Format format, const vk::ImageTiling tiling, const vk::ImageUsageFlags usage, const vk::MemoryPropertyFlags properties, const VmaMemoryUsage vmaUsage) const;
    void transitionImageLayout(const Image& image, const vk::ImageSubresourceRange& subresourceRange, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout) const;
    void destroy(const Image& image);

    vk::CommandBuffer beginGraphicsCommands() const;
    void endGraphicsCommands(const vk::CommandBuffer buffer) const;

private:
    bool isDeviceSuitable(const vk::PhysicalDevice device) const;

    void createInstance();
    void createDebugMessenger();
    void createSurface(GLFWwindow* window);
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    void createCommandPool();

    vk::Instance _instance;
    vk::PhysicalDevice _physical;
    vk::Device _logical;
    VmaAllocator _allocator;
    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies = {std::nullopt, std::nullopt};
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::CommandPool _commandPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
