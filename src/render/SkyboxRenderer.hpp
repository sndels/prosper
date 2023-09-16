#ifndef PROSPER_RENDER_SKYBOX_RENDERER_HPP
#define PROSPER_RENDER_SKYBOX_RENDERER_HPP

#include "../gfx/Device.hpp"
#include "../gfx/Swapchain.hpp"
#include "../scene/Camera.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class SkyboxRenderer
{
  public:
    SkyboxRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const World::DSLayouts &worldDSLayouts);
    ~SkyboxRenderer();

    SkyboxRenderer(const SkyboxRenderer &other) = delete;
    SkyboxRenderer(SkyboxRenderer &&other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(SkyboxRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    void record(
        vk::CommandBuffer cb, const World &world,
        const RecordInOut &inOutTargets, uint32_t nextFrame,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, const RecordInOut &inOutTargets) const;

    struct Attachments
    {
        vk::RenderingAttachmentInfo color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets) const;

    void destroyGraphicsPipelines();

    void createGraphicsPipelines(const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_SKYBOX_RENDERER_HPP
