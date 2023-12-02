#ifndef PROSPER_RENDER_GBUFFER_RENDERER_HPP
#define PROSPER_RENDER_GBUFFER_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DebugDrawTypes.hpp"
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
    ImageHandle depth;
};

class GBufferRenderer
{
  public:
    GBufferRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);
    ~GBufferRenderer();

    GBufferRenderer(const GBufferRenderer &other) = delete;
    GBufferRenderer(GBufferRenderer &&other) = delete;
    GBufferRenderer &operator=(const GBufferRenderer &other) = delete;
    GBufferRenderer &operator=(GBufferRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void drawUi();

    [[nodiscard]] GBufferRendererOutput record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, uint32_t nextFrame, Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void destroyGraphicsPipeline();

    [[nodiscard]] GBufferRendererOutput createOutputs(
        const vk::Extent2D &size) const;

    struct Attachments
    {
        wheels::StaticArray<vk::RenderingAttachmentInfo, 2> color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const GBufferRendererOutput &output) const;

    void recordBarriers(
        vk::CommandBuffer cb, const GBufferRendererOutput &output) const;

    void createGraphicsPipelines(
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{{}};
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_GBUFFER_RENDERER_HPP
