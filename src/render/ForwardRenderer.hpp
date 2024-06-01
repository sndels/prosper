#ifndef PROSPER_RENDER_FORWARD_RENDERER_HPP
#define PROSPER_RENDER_FORWARD_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DrawType.hpp"
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
    ForwardRenderer() noexcept = default;
    ~ForwardRenderer();

    ForwardRenderer(const ForwardRenderer &other) = delete;
    ForwardRenderer(ForwardRenderer &&other) = delete;
    ForwardRenderer &operator=(const ForwardRenderer &other) = delete;
    ForwardRenderer &operator=(ForwardRenderer &&other) = delete;

    struct InputDSLayouts
    {
        vk::DescriptorSetLayout camera;
        vk::DescriptorSetLayout lightClusters;
        const WorldDSLayouts &world;
    };
    void init(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc, RenderResources *resources,
        const InputDSLayouts &dsLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    struct OpaqueOutput
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    [[nodiscard]] OpaqueOutput recordOpaque(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        MeshletCuller *meshletCuller, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea,
        const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
        uint32_t nextFrame, bool applyIbl, DrawType drawType,
        SceneStats *sceneStats, Profiler *profiler);

    struct TransparentInOut
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    void recordTransparent(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        MeshletCuller *meshletCuller, const World &world, const Camera &cam,
        const TransparentInOut &inOutTargets,
        const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
        uint32_t nextFrame, DrawType drawType, SceneStats *sceneStats,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);

    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame, bool transparents,
        const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats);

    void destroyGraphicsPipelines();
    void createGraphicsPipelines(const InputDSLayouts &dsLayouts);

    struct Options
    {
        bool transparents{false};
        bool ibl{false};
        DrawType drawType{DrawType::Default};
    };
    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        MeshletCuller *meshletCuller, const World &world, const Camera &cam,
        uint32_t nextFrame, const RecordInOut &inOutTargets,
        const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
        const Options &options, SceneStats *sceneStats, Profiler *profiler,
        const char *debugName);
    void recordBarriers(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const RecordInOut &inOutTargets,
        const LightClusteringOutput &lightClusters,
        const MeshletCullerOutput &cullerOutput,
        BufferHandle inOutDrawStats) const;

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

    bool _initialized{false};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;
    wheels::Optional<ShaderReflection> _meshReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> _pipelines;

    vk::DescriptorSetLayout _meshSetLayout;
    // Separate sets for transparents and opaque
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT * 2> _meshSets{
        VK_NULL_HANDLE};
};

#endif // PROSPER_RENDER_FORWARD_RENDERER_HPP
