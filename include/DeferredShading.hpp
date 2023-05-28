#ifndef PROSPER_DEFERRED_SHADING_HPP
#define PROSPER_DEFERRED_SHADING_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "GBufferRenderer.hpp"
#include "LightClustering.hpp"
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

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const World::DSLayouts &world;
    };
    DeferredShading(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        const InputDSLayouts &dsLayouts);

    ~DeferredShading();

    DeferredShading(const DeferredShading &other) = delete;
    DeferredShading(DeferredShading &&other) = delete;
    DeferredShading &operator=(const DeferredShading &other) = delete;
    DeferredShading &operator=(DeferredShading &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

    void drawUi();

    struct Input
    {
        const GBufferRenderer::Output &gbuffer;
        const LightClustering::Output &lightClusters;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const Input &input, uint32_t nextFrame, Profiler *profiler);

  private:
    [[nodiscard]] wheels::Optional<ShaderReflection> compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void recordBarriers(
        vk::CommandBuffer cb, const Input &input, const Output &output) const;

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc,
        const ShaderReflection &reflection);
    struct BoundImages
    {
        ImageHandle albedoRoughness;
        ImageHandle normalMetalness;
        ImageHandle depth;
        ImageHandle illumination;
    };
    void updateDescriptorSet(uint32_t nextFrame, const BoundImages &images);
    void createPipeline(const InputDSLayouts &dsLayouts);

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
