#ifndef PROSPER_RENDER_FORWARD_RENDERER_HPP
#define PROSPER_RENDER_FORWARD_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>

class ForwardRenderer
{
  public:
    enum class DrawType : uint32_t
    {
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const WorldDSLayouts &world;
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
        ImageHandle velocity;
        ImageHandle depth;
    };
    [[nodiscard]] OpaqueOutput recordOpaque(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        const LightClusteringOutput &lightClusters, uint32_t nextFrame,
        bool applyIbl, Profiler *profiler);

    struct TransparentInOut
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    void recordTransparent(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam,
        const TransparentInOut &inOutTargets,
        const LightClusteringOutput &lightClusters, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void destroyGraphicsPipelines();
    void createGraphicsPipelines(const InputDSLayouts &dsLayouts);

    struct Options
    {
        bool transparents{false};
        bool ibl{false};
    };
    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, uint32_t nextFrame,
        const RecordInOut &inOutTargets,
        const LightClusteringOutput &lightClusters, const Options &options,
        Profiler *profiler, const char *debugName);
    void recordBarriers(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const RecordInOut &inOutTargets,
        const LightClusteringOutput &lightClusters) const;

    struct Attachments
    {
        wheels::InlineArray<vk::RenderingAttachmentInfo, 2> color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets, bool transparents) const;

    static vk::Rect2D getRenderArea(
        const RenderResources &resources,
        const ForwardRenderer::RecordInOut &inOutTargets);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> _pipelines;

    DrawType _drawType{DrawType::Default};
};

#endif // PROSPER_RENDER_FORWARD_RENDERER_HPP
