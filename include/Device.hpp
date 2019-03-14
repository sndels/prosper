#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#include <vulkan/vulkan.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
    vk::DeviceMemory memory;
};

class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device& other) = delete;
    Device operator=(const Device& other) = delete;

    void init(GLFWwindow* window);

    vk::Instance instance();
    vk::PhysicalDevice physical();
    vk::Device logical();
    vk::SurfaceKHR surface();
    vk::CommandPool commandPool();
    vk::Queue graphicsQueue();
    vk::Queue presentQueue();
    const QueueFamilies& queueFamilies() const;

    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
    void copyBuffer(const Buffer& src, const Buffer& dst, vk::DeviceSize size);

private:
    bool isDeviceSuitable(vk::PhysicalDevice device);

    void createInstance();
    void createDebugMessenger();
    void createSurface(GLFWwindow* window);
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    vk::Instance _instance;
    vk::PhysicalDevice _physical;
    vk::Device _logical;
    vk::SurfaceKHR _surface;

    QueueFamilies _queueFamilies = {std::nullopt, std::nullopt};
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::CommandPool _commandPool;

    vk::DebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
