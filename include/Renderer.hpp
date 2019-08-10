#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class Renderer {
public:
    struct Pipelines {
        vk::Pipeline pbr;
        vk::Pipeline pbrAlphaBlend;
        vk::Pipeline skybox;
    };

    struct PipelineLayouts {
        vk::PipelineLayout pbr;
        vk::PipelineLayout skybox;
    };

    Renderer() = default;
    ~Renderer();

    // Two-part initialization simplifies things in App
    void init(std::shared_ptr<Device> device);
    void createSwapchainRelated(const SwapchainConfig& swapConfig, const vk::DescriptorSetLayout camDSLayout, const World::DSLayouts& worldDSLayouts);
    void destroySwapchainRelated();

    vk::Semaphore imageAvailable(const uint32_t frame) const;
    vk::RenderPass outputRenderpass() const;

    std::array<vk::Semaphore, 1> drawFrame(const World& world, const Camera& cam, const Swapchain& swapchain, const uint32_t nextImage) const;

private:
    // These also need to be recreated with Swapchain as they depend on swapconfig
    void createRenderPass(const SwapchainConfig& swapConfig);
    void createFramebuffer(const SwapchainConfig& swapConfig);
    void createGraphicsPipelines(const SwapchainConfig& swapConfig, const vk::DescriptorSetLayout camDSLayout, const World::DSLayouts& worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig& swapConfig);

    void createSemaphores(const uint32_t concurrentFrameCount);

    void updateUniformBuffers(const World& world, const Camera& cam, const uint32_t nextImage) const;
    void recordCommandBuffer(const World& world, const Camera& cam, const Swapchain& swapchain, const uint32_t nextImage) const;
    void recordModelInstances(const vk::CommandBuffer buffer, const uint32_t nextImage, const std::vector<Scene::ModelInstance>& instances, const std::function<bool(const Mesh&)>& cullMesh) const;

    std::shared_ptr<Device> _device = nullptr;

    PipelineLayouts _pipelineLayouts;
    Pipelines _pipelines;

    vk::RenderPass _renderpass;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
    vk::ImageBlit _fboToSwap;
    Image _colorImage = {nullptr, nullptr, {}, nullptr};
    Image _depthImage = {nullptr, nullptr, {}, nullptr};

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;

};

#endif // PROSPER_RENDERER_HPP
