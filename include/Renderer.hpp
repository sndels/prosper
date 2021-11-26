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

    vk::CommandBuffer execute(
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const SwapchainConfig &swapConfig);
    void createRenderPass();
    void createFramebuffer(const SwapchainConfig &swapConfig);
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig,
        const vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    void updateUniformBuffers(
        const World &world, const Camera &cam, const uint32_t nextImage) const;
    vk::CommandBuffer recordCommandBuffer(
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        const uint32_t nextImage) const;
    void recordModelInstances(
        const vk::CommandBuffer buffer, const uint32_t nextImage,
        const std::vector<Scene::ModelInstance> &instances,
        const std::function<bool(const Mesh &)> &cullMesh) const;

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::RenderPass _renderpass;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
};

#endif // PROSPER_RENDERER_HPP
