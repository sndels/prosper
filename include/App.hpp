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

    vk::DescriptorSetLayout _vkCameraDescriptorSetLayout;
    vk::PipelineLayout _vkGraphicsPipelineLayout;

    vk::DescriptorPool _vkDescriptorPool;
    std::vector<vk::DescriptorSet> _vkCameraDescriptorSets;

    // TODO: Make these dynamic
    std::vector<Buffer> _transformBuffers;

    vk::RenderPass _vkRenderPass;
    vk::Pipeline _vkGraphicsPipeline;

    std::vector<vk::CommandBuffer> _vkCommandBuffers;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_APP_HPP
