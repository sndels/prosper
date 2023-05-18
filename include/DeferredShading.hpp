#ifndef PROSPER_DEFERRED_SHADING_HPP
#define PROSPER_DEFERRED_SHADING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "GBufferRenderer.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"
#include "World.hpp"

class DeferredShading
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    DeferredShading(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~DeferredShading();

    DeferredShading(const DeferredShading &other) = delete;
    DeferredShading(DeferredShading &&other) = delete;
    DeferredShading &operator=(const DeferredShading &other) = delete;
    DeferredShading &operator=(DeferredShading &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const GBufferRenderer::Output &gbuffer, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void destroyPipelines();

    void createDescriptorSets();
    struct BoundImages
    {
        ImageHandle albedoRoughness;
        ImageHandle normalMetalness;
        ImageHandle depth;
        ImageHandle illumination;
    };
    void updateDescriptorSet(uint32_t nextFrame, const BoundImages &images);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::Sampler _depthSampler;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_DEFERRED_SHADING_HPP
