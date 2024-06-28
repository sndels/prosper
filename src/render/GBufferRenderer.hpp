#ifndef PROSPER_RENDER_GBUFFER_RENDERER_HPP
#define PROSPER_RENDER_GBUFFER_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DrawType.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

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
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    [[nodiscard]] GBufferRendererOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        MeshletCuller *meshletCuller, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, BufferHandle inOutDrawStats,
        DrawType drawType, uint32_t nextFrame, SceneStats *sceneStats);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame,
        const MeshletCullerOutput &cullerOutput, BufferHandle inOutDrawStats);

    void destroyGraphicsPipeline();

    void createGraphicsPipelines(
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<ShaderReflection> m_meshReflection;
    wheels::Optional<ShaderReflection> m_fragReflection;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;

    vk::DescriptorSetLayout m_meshSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> m_meshSets{
        VK_NULL_HANDLE};
};

#endif // PROSPER_RENDER_GBUFFER_RENDERER_HPP
