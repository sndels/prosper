#ifndef PROSPER_TRANSPARENTS_RENDERER_HPP
#define PROSPER_TRANSPARENTS_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class TransparentsRenderer
{
  public:
    struct PipelineLayouts
    {
        vk::PipelineLayout pbr;
        vk::PipelineLayout skybox;
    };

    TransparentsRenderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~TransparentsRenderer();

    TransparentsRenderer(const TransparentsRenderer &other) = delete;
    TransparentsRenderer(TransparentsRenderer &&other) = delete;
    TransparentsRenderer &operator=(const TransparentsRenderer &other) = delete;
    TransparentsRenderer &operator=(TransparentsRenderer &&other) = delete;

    void recompileShaders(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    [[nodiscard]] vk::CommandBuffer recordCommandBuffer(
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        uint32_t nextImage) const;

  private:
    bool compileShaders(const World::DSLayouts &worldDSLayouts);

    void destroySwapchainRelated();
    void destroyGraphicsPipeline();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createAttachments();
    void createGraphicsPipeline(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    std::array<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;

    vk::RenderingAttachmentInfo _colorAttachment;
    vk::RenderingAttachmentInfo _depthAttachment;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_TRANSPARENTS_RENDERER_HPP
