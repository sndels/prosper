#ifndef PROSPER_RENDER_GBUFFER_RENDERER_HPP
#define PROSPER_RENDER_GBUFFER_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DrawType.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"
#include "render/HierarchicalDepthDownsampler.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

struct GBufferRendererOutput
{
    ImageHandle albedoRoughness;
    ImageHandle normalMetalness;
    ImageHandle velocity;
    ImageHandle depth;
};

class GBufferRenderer
{
  public:
    GBufferRenderer() noexcept = default;
    ~GBufferRenderer();

    GBufferRenderer(const GBufferRenderer &other) = delete;
    GBufferRenderer(GBufferRenderer &&other) = delete;
    GBufferRenderer &operator=(const GBufferRenderer &other) = delete;
    GBufferRenderer &operator=(GBufferRenderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts, MeshletCuller *meshletCuller,
        HierarchicalDepthDownsampler *hierarchicalDepthDownsampler);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] GBufferRendererOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        BufferHandle inOutDrawStats, DrawType drawType, uint32_t nextFrame,
        DrawStats *drawStats);
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

    void destroyGraphicsPipeline();

    void createGraphicsPipelines(
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    struct RecordInOut
    {
        BufferHandle inDataBuffer;
        BufferHandle inArgumentBuffer;
        BufferHandle inOutDrawStats;
        ImageHandle outAlbedoRoughness;
        ImageHandle outNormalMetalness;
        ImageHandle outVelocity;
        ImageHandle outDepth;
    };
    void recordDraw(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const vk::Rect2D &renderArea,
        uint32_t nextFrame, const RecordInOut &inputsOutputs, DrawType drawType,
        bool isSecondPhase, DrawStats *drawStats);

    bool m_initialized{false};

    MeshletCuller *m_meshletCuller{nullptr};
    HierarchicalDepthDownsampler *m_hierarchicalDepthDownsampler{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<ShaderReflection> m_meshReflection;
    wheels::Optional<ShaderReflection> m_fragReflection;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;

    vk::DescriptorSetLayout m_meshSetLayout;
    // Two sets per frame for the two pass culled draw
    static const uint32_t sDescriptorSetCount = MAX_FRAMES_IN_FLIGHT * 2;
    wheels::StaticArray<vk::DescriptorSet, sDescriptorSetCount> m_meshSets{
        VK_NULL_HANDLE};

    ImageHandle m_previousHierarchicalDepth;
};

#endif // PROSPER_RENDER_GBUFFER_RENDERER_HPP
