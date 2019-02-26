#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <vector>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class App {
public:
    void run();

private:
    void initWindow();
    void cleanup();

    // Vk -stuff
    void initVulkan();
    void createInstance();
    void createDebugMessenger();
    void createSurface();
    bool isDeviceSuitable(VkPhysicalDevice device);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    VkShaderModule createShaderModule(const std::vector<char>& spv);

    void mainLoop();

    VkInstance _vkInstance;
    VkPhysicalDevice _vkPhysicalDevice;
    VkDevice _vkDevice;
    VkQueue _graphicsQueue;
    VkQueue _presentQueue;
    VkSurfaceKHR _vkSurface;
    VkSwapchainKHR _vkSwapchain;
    std::vector<VkImage> _vkSwapchainImages;
    VkFormat _vkSwapchainImageFormat;
    VkExtent2D _vkSwapchainExtent;
    std::vector<VkImageView> _vkSwapchainImageViews;
    std::vector<VkFramebuffer> _vkSwapchainFramebuffers;
    VkRenderPass _vkRenderPass;
    VkPipelineLayout _vkGraphicsPipelineLayout;
    VkPipeline _vkGraphicsPipeline;
    VkCommandPool _vkCommandPool;
    std::vector<VkCommandBuffer> _vkCommandBuffers;
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    std::vector<VkFence> _inFlightFences;
    size_t _currentFrame = 0;

    VkDebugUtilsMessengerEXT _debugMessenger;

    GLFWwindow* _window;
};

#endif // PROSPER_APP_HPP
