#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <vector>

#include "Device.hpp"

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

class App {
public:
    App();
    ~App();

    App(const App& other) = delete;
    App operator=(const App& other) = delete;

    void init();
    void run();

private:
    void initWindow();

    void initVulkan();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    VkShaderModule createShaderModule(const std::vector<char>& spv);

    VkSwapchainKHR _vkSwapchain;
    std::vector<VkImage> _vkSwapchainImages;
    VkFormat _vkSwapchainImageFormat;
    VkExtent2D _vkSwapchainExtent;
    std::vector<VkImageView> _vkSwapchainImageViews;
    std::vector<VkFramebuffer> _vkSwapchainFramebuffers;

    VkRenderPass _vkRenderPass;
    VkPipelineLayout _vkGraphicsPipelineLayout;
    VkPipeline _vkGraphicsPipeline;
    std::vector<VkCommandBuffer> _vkCommandBuffers;
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    std::vector<VkFence> _inFlightFences;
    size_t _currentFrame;

    // Destruct device after other potential vulkan-resources
    Device _device;
    GLFWwindow* _window;
};

#endif // PROSPER_APP_HPP
