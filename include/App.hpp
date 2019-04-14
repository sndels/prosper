#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include <functional>
#include <optional>
#include <vector>

#include "Camera.hpp"
#include "Device.hpp"
#include "Mesh.hpp"
#include "Swapchain.hpp"
#include "Texture.hpp"
#include "Window.hpp"
#include "World.hpp"

class App {
public:
    struct Pipelines {
        vk::Pipeline pbr;
        vk::Pipeline pbrAlphaBlend;
    };

    App() = default;
    ~App();

    App(const App& other) = delete;
    App& operator=(const App& other) = delete;

    void init();
    void run();

private:
    void recreateSwapchainAndRelated();
    void destroySwapchainRelated();

    void createUniformBuffers(const uint32_t swapImageCount);
    void createDescriptorPool(const uint32_t swapImageCount);
    void createSemaphores(const uint32_t concurrentFrameCount);

    // These also need to be recreated with Swapchain as they depend on swapconfig / swapchain
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createGraphicsPipeline(const SwapchainConfig& swapConfig);
    void createCommandBuffers(const SwapchainConfig& swapConfig);

    void drawFrame();
    void updateUniformBuffers(const uint32_t nextImage);
    void recordCommandBuffer(const uint32_t nextImage);
    void recordModelInstances(const vk::CommandBuffer buffer, const uint32_t nextImage, const std::vector<Scene::ModelInstance>& instances, const std::function<bool(const Mesh&)>& cullMesh);

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources
    Swapchain _swapchain;
    World _world;
    Camera _cam;

    vk::PipelineLayout _vkGraphicsPipelineLayout;

    vk::DescriptorPool _vkDescriptorPool;

    vk::RenderPass _vkRenderPass;
    Pipelines _pipelines;

    std::vector<vk::CommandBuffer> _vkCommandBuffers;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_APP_HPP
