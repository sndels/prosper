#ifndef PROSPER_DEBUG_RENDERER_HPP
#define PROSPER_DEBUG_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class DebugRenderer
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    DebugRenderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout);
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer &other) = delete;
    DebugRenderer(DebugRenderer &&other) = delete;
    DebugRenderer &operator=(const DebugRenderer &other) = delete;
    DebugRenderer &operator=(DebugRenderer &&other) = delete;

    void recompileShaders(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout);

    void recreate(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout);

    void record(
        vk::CommandBuffer cb, const Camera &cam, const vk::Rect2D &renderArea,
        uint32_t nextImage, Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders();

    void destroySwapchainRelated();
    void destroyGraphicsPipeline();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createBuffers(const SwapchainConfig &swapConfig);
    void createDescriptorSets(const uint32_t swapImageCount);
    void createAttachments();
    void createGraphicsPipeline(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    std::array<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{};

    vk::RenderingAttachmentInfo _colorAttachment;
    vk::RenderingAttachmentInfo _depthAttachment;

    vk::DescriptorSetLayout _linesDSLayout;
    std::vector<vk::DescriptorSet> _linesDescriptorSets;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_DEBUG_RENDERER_HPP
