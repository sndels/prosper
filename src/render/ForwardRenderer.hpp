#ifndef PROSPER_RENDER_FORWARD_RENDERER_HPP
#define PROSPER_RENDER_FORWARD_RENDERER_HPP

#include "gfx/Fwd.hpp"
#include "gfx/ShaderReflection.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"
#include "utils/Fwd.hpp"

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
        wheels::ScopedScratch scopeAlloc, const InputDSLayouts &dsLayouts,
        MeshletCuller *meshletCuller,
        HierarchicalDepthDownsampler *hierarchicalDepthDownsampler);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const InputDSLayouts &dsLayouts);

    void startFrame();

    struct OpaqueOutput
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    [[nodiscard]] OpaqueOutput recordOpaque(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        const LightClusteringOutput &lightClusters, BufferHandle inOutDrawStats,
        uint32_t nextFrame, bool applyIbl, DrawType drawType,
        DrawStats *drawStats);

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
        uint32_t nextFrame, DrawType drawType, DrawStats *drawStats);

    void releasePreserved();

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);

    struct DescriptorSetBuffers
    {
        BufferHandle dataBuffer;
        BufferHandle drawStats;
    };
    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSet ds,
        const DescriptorSetBuffers &buffers) const;

    void destroyGraphicsPipelines();
    void createGraphicsPipelines(const InputDSLayouts &dsLayouts);

    struct Options
    {
        bool transparents{false};
        bool ibl{false};
        bool secondPhase{false};
        DrawType drawType{DrawType::Default};
    };
    struct RecordInOut
    {
        ImageHandle inOutIllumination;
        ImageHandle inOutVelocity;
        ImageHandle inOutDepth;
        BufferHandle inOutDrawStats;
        BufferHandle inDataBuffer;
        BufferHandle inArgumentBuffer;
    };
    void recordDraw(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, uint32_t nextFrame,
        const RecordInOut &inputsOutputs,
        const LightClusteringOutput &lightClusters, const Options &options,
        DrawStats *drawStats, const char *debugName);

    bool m_initialized{false};

    MeshletCuller *m_meshletCuller{nullptr};
    HierarchicalDepthDownsampler *m_hierarchicalDepthDownsampler{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<ShaderReflection> m_meshReflection;
    wheels::Optional<ShaderReflection> m_fragReflection;

    vk::PipelineLayout m_pipelineLayout;
    wheels::StaticArray<vk::Pipeline, 2> m_pipelines;

    vk::DescriptorSetLayout m_meshSetLayout;
    uint32_t m_nextFrameRecord{0};
    // Separate sets for transparents and opaque, and for the two culling phases
    // for each
    static const uint32_t sDescriptorSetCount = MAX_FRAMES_IN_FLIGHT * 2 * 2;
    wheels::StaticArray<vk::DescriptorSet, sDescriptorSetCount> m_meshSets{
        VK_NULL_HANDLE};

    ImageHandle m_previousHierarchicalDepth;
};

#endif // PROSPER_RENDER_FORWARD_RENDERER_HPP
