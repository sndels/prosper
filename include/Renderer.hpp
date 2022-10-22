#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include <functional>

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

class Renderer
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    Renderer(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer(Renderer &&other) = delete;
    Renderer &operator=(const Renderer &other) = delete;
    Renderer &operator=(Renderer &&other) = delete;

    void recompileShaders(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreateSwapchainRelated(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    void record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, uint32_t nextImage, bool transparents,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(const World::DSLayouts &worldDSLayouts);

    void destroySwapchainRelated();
    void destroyGraphicsPipelines();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const SwapchainConfig &swapConfig);
    void createAttachments();
    void createGraphicsPipelines(
        const SwapchainConfig &swapConfig, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    std::array<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{};

    std::array<vk::RenderingAttachmentInfo, 2> _colorAttachments{};
    std::array<vk::RenderingAttachmentInfo, 2> _depthAttachments{};

    vk::PipelineLayout _pipelineLayout;
    std::array<vk::Pipeline, 2> _pipelines{};

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDERER_HPP
