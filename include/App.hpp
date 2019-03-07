#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include <optional>
#include <vector>

#include "Device.hpp"
#include "Swapchain.hpp"
#include "Window.hpp"

class App {
public:
    App() = default;
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

    Window _window;
    Device _device;
    Swapchain _swapchain;

    VkRenderPass _vkRenderPass = VK_NULL_HANDLE;
    VkPipelineLayout _vkGraphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline _vkGraphicsPipeline = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> _vkCommandBuffers;
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
};

#endif // PROSPER_APP_HPP
