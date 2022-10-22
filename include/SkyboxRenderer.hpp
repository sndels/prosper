#ifndef PROSPER_SKYBOXRENDERER_HPP
#define PROSPER_SKYBOXRENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
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
    SkyboxRenderer(SkyboxRenderer &&other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(SkyboxRenderer &&other) = delete;

    void recompileShaders(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);

    void record(
        vk::CommandBuffer cb, const World &world, const vk::Rect2D &renderArea,
        uint32_t nextImage, Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders();

    void destroySwapchainRelated();
    void destroyGraphicsPipelines();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createAttachments();
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    std::array<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;

    vk::RenderingAttachmentInfo _colorAttachment;
    vk::RenderingAttachmentInfo _depthAttachment;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_SKYBOXRENDERER_HPP
