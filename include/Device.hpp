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
    Device() = default;
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

    VkInstance _instance = VK_NULL_HANDLE;
    VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
    VkDevice _device = VK_NULL_HANDLE;
    VkSurfaceKHR _surface = VK_NULL_HANDLE;

    QueueFamilies _queueFamilies = {std::nullopt, std::nullopt};
    VkQueue _graphicsQueue = VK_NULL_HANDLE;
    VkQueue _presentQueue = VK_NULL_HANDLE;

    VkCommandPool _commandPool = VK_NULL_HANDLE;

    VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
};

#endif // PROSPER_DEVICE_HPP
