#ifndef PROSPER_GBUFFER_RENDERER_HPP
#define PROSPER_GBUFFER_RENDERER_HPP

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class GBufferRenderer
{
  public:
    GBufferRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const vk::Extent2D &renderExtent,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~GBufferRenderer();

    GBufferRenderer(const GBufferRenderer &other) = delete;
    GBufferRenderer(GBufferRenderer &&other) = delete;
    GBufferRenderer &operator=(const GBufferRenderer &other) = delete;
    GBufferRenderer &operator=(GBufferRenderer &&other) = delete;

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
        const vk::Rect2D &renderArea, uint32_t nextImage,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void destroyViewportRelated();
    void destroyGraphicsPipeline();

    void createOutputs(const vk::Extent2D &renderExtent);
    void createAttachments();
    void createGraphicsPipelines(
        const vk::Extent2D &renderExtent, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{{}};

    wheels::StaticArray<vk::RenderingAttachmentInfo, 2> _colorAttachments{{}};
    vk::RenderingAttachmentInfo _depthAttachment;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_GBUFFER_RENDERER_HPP
