#ifndef PROSPER_SKYBOXRENDERER_HPP
#define PROSPER_SKYBOXRENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class SkyboxRenderer
{
  public:
    SkyboxRenderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);
    ~SkyboxRenderer();

    SkyboxRenderer(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);

    vk::Semaphore imageAvailable(const size_t frame) const;
    vk::RenderPass outputRenderpass() const;

    vk::CommandBuffer execute(
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createRenderPass();
    void createFramebuffer(const SwapchainConfig &swapConfig);
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    void updateUniformBuffers(
        const World &world, const Camera &cam, const uint32_t nextImage) const;
    vk::CommandBuffer recordCommandBuffer(
        const World &world, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::RenderPass _renderpass;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
};

#endif // PROSPER_SKYBOXRENDERER_HPP
