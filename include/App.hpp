#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <vector>

#include "Device.hpp"
#include "Swapchain.hpp"

class App {
public:
    App();
    ~App();

    App(const App& other) = delete;
    App operator=(const App& other) = delete;

    void init();
    void run();

private:
    // After Device and before Swapchain
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createGraphicsPipeline(const SwapchainConfig& swapConfig);

    // After Swapchain
    void createCommandBuffers();
    void createSemaphores();

    void drawFrame();

    VkRenderPass _vkRenderPass;
    VkPipelineLayout _vkGraphicsPipelineLayout;
    VkPipeline _vkGraphicsPipeline;
    std::vector<VkCommandBuffer> _vkCommandBuffers;
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;

    Swapchain _swapchain;
    // Destruct device after other potential vulkan-resources
    Device _device;
    GLFWwindow* _window;
};

#endif // PROSPER_APP_HPP
