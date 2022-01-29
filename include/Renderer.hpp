#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class Renderer
{
  public:
    Renderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer &operator=(const Renderer &other) = delete;

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    vk::CommandBuffer recordCommandBuffer(
        const Scene &scene, const Camera &cam, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const SwapchainConfig &swapConfig);
    void createAttachments();
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    void recordModelInstances(
        const vk::CommandBuffer buffer, const uint32_t nextImage,
        const std::vector<Scene::ModelInstance> &instances,
        const std::function<bool(const Mesh &)> &shouldRender) const;

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::RenderingAttachmentInfoKHR _colorAttachment;
    vk::RenderingAttachmentInfoKHR _depthAttachment;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_RENDERER_HPP
