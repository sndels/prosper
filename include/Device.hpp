#ifndef PROSPER_DEVICE_HPP
#define PROSPER_DEVICE_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <vector>

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class Device {
public:
    Device();
    ~Device();

    Device(const Device& other) = delete;
    Device operator=(const Device& other) = delete;

    void init(GLFWwindow* window);

    VkInstance instance();
    VkPhysicalDevice physicalDevice();
    VkDevice handle();
    VkSurfaceKHR surface();
    VkCommandPool commandPool();
    VkQueue graphicsQueue();
    VkQueue presentQueue();
    const QueueFamilies& queueFamilies() const;

private:
    bool isDeviceSuitable(VkPhysicalDevice device);

    void createInstance();
    void createDebugMessenger();
    void createSurface(GLFWwindow* window);
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    VkInstance _instance;
    VkPhysicalDevice _physicalDevice;
    VkDevice _device;
    VkSurfaceKHR _surface;

    QueueFamilies _queueFamilies;
    VkQueue _graphicsQueue;
    VkQueue _presentQueue;

    VkCommandPool _commandPool;

    VkDebugUtilsMessengerEXT _debugMessenger;
};

#endif // PROSPER_DEVICE_HPP
