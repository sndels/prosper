#ifndef PROSPER_RENDER_FORWARD_RENDERER_HPP
#define PROSPER_RENDER_FORWARD_RENDERER_HPP

#include "../gfx/Device.hpp"
#include "../gfx/Swapchain.hpp"
#include "../scene/Camera.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "LightClustering.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class ForwardRenderer
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
    ForwardRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const InputDSLayouts &dsLayouts);
    ~ForwardRenderer();

    ForwardRenderer(const ForwardRenderer &other) = delete;
    ForwardRenderer(ForwardRenderer &&other) = delete;
    ForwardRenderer &operator=(const ForwardRenderer &other) = delete;
    ForwardRenderer &operator=(ForwardRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

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
        bool applyIbl, Profiler *profiler);

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

    struct Options
    {
        bool transparents{false};
        bool ibl{false};
    };
    void record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        uint32_t nextFrame, const RecordInOut &inOutTargets,
        const LightClustering::Output &lightClusters, const Options &options,
        Profiler *profiler, const char *debugName);
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
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> _pipelines{{}};

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDER_FORWARD_RENDERER_HPP
