#ifndef PROSPER_SKYBOXRENDERER_HPP
#define PROSPER_SKYBOXRENDERER_HPP

#include "Camera.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

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

    void destroyGraphicsPipelines();

    void createGraphicsPipelines(const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_SKYBOXRENDERER_HPP
