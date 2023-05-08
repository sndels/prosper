#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class Renderer
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    Renderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const vk::Extent2D &renderExtent,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer(Renderer &&other) = delete;
    Renderer &operator=(const Renderer &other) = delete;
    Renderer &operator=(Renderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, const vk::Extent2D &renderExtent,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void recreate(
        const vk::Extent2D &renderExtent, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    void record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, uint32_t nextImage, bool transparents,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void destroySwapchainRelated();
    void destroyGraphicsPipelines();
    // These also need to be recreated with Swapchain as they depend on
    // swapconfig
    void createOutputs(const vk::Extent2D &renderExtent);
    void createAttachments();
    void createGraphicsPipelines(
        const vk::Extent2D &renderExtent, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{{}};

    wheels::StaticArray<vk::RenderingAttachmentInfo, 2> _colorAttachments{{}};
    wheels::StaticArray<vk::RenderingAttachmentInfo, 2> _depthAttachments{{}};

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> _pipelines{{}};

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDERER_HPP
