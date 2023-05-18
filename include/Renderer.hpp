#ifndef PROSPER_RENDERER_HPP
#define PROSPER_RENDERER_HPP

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "LightClustering.hpp"
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

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const World::DSLayouts &world;
    };
    Renderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const InputDSLayouts &dsLayouts);
    ~Renderer();

    Renderer(const Renderer &other) = delete;
    Renderer(Renderer &&other) = delete;
    Renderer &operator=(const Renderer &other) = delete;
    Renderer &operator=(Renderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts);

    void drawUi();

    struct OpaqueOutput
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    [[nodiscard]] OpaqueOutput recordOpaque(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea,
        const LightClustering::Output &lightClusters, uint32_t nextFrame,
        Profiler *profiler);

    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    void recordTransparent(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const RecordInOut &inOutTargets,
        const LightClustering::Output &lightClusters, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void destroyGraphicsPipelines();
    void createGraphicsPipelines(const InputDSLayouts &dsLayouts);

    void record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        uint32_t nextFrame, const RecordInOut &inOutTargets,
        const LightClustering::Output &lightClusters, bool transparents);
    void recordBarriers(
        vk::CommandBuffer cb, const RecordInOut &inOutTargets,
        const LightClustering::Output &lightClusters) const;

    struct Attachments
    {
        vk::RenderingAttachmentInfo color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets, bool transparents) const;

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{{}};

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> _pipelines{{}};

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDERER_HPP
