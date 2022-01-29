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

    vk::CommandBuffer recordCommandBuffer(
        const World &world, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createAttachments();
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::RenderingAttachmentInfoKHR _colorAttachment;
    vk::RenderingAttachmentInfoKHR _depthAttachment;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_SKYBOXRENDERER_HPP
