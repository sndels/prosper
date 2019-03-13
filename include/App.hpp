#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include <optional>
#include <vector>

#include "Camera.hpp"
#include "Device.hpp"
#include "Mesh.hpp"
#include "Swapchain.hpp"
#include "Window.hpp"

struct Transforms {
    glm::mat4 modelToClip;
};

class App {
public:
    App() = default;
    ~App();

    App(const App& other) = delete;
    App operator=(const App& other) = delete;

    void init();
    void run();

private:
    // Recreates swapchain and resources tied to it
    void recreateSwapchainAndRelated();
    // Destroys resources dependent on current swapchain
    void destroySwapchainRelated();

    // Before pipeline
    void createDescriptorSetLayout();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    // These need to be recreated with Swapchain
    // Before swapchain
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createGraphicsPipeline(const SwapchainConfig& swapConfig);
    // After swapchain
    void createCommandBuffers();

    void createSemaphores();

    void drawFrame();
    void updateUniformBuffer(uint32_t nextImage);
    void recordCommandBuffer(uint32_t nextImage);

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources
    Swapchain _swapchain;
    std::vector<Mesh> _meshes;
    Camera _cam;

    VkDescriptorSetLayout _vkDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _vkGraphicsPipelineLayout = VK_NULL_HANDLE;

    VkDescriptorPool _vkDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> _vkDescriptorSets;

    std::vector<Buffer> _transformBuffers;

    VkRenderPass _vkRenderPass = VK_NULL_HANDLE;
    VkPipeline _vkGraphicsPipeline = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> _vkCommandBuffers;

    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_APP_HPP
