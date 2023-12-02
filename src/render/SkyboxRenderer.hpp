#ifndef PROSPER_RENDER_SKYBOX_RENDERER_HPP
#define PROSPER_RENDER_SKYBOX_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class SkyboxRenderer
{
  public:
    SkyboxRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const WorldDSLayouts &worldDSLayouts);
    ~SkyboxRenderer();

    SkyboxRenderer(const SkyboxRenderer &other) = delete;
    SkyboxRenderer(SkyboxRenderer &&other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(SkyboxRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const WorldDSLayouts &worldDSLayouts);

    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    void record(
        vk::CommandBuffer cb, const World &world, const Camera &camera,
        const RecordInOut &inOutTargets, Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, const RecordInOut &inOutTargets) const;

    struct Attachments
    {
        vk::RenderingAttachmentInfo color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets) const;

    void destroyGraphicsPipelines();

    void createGraphicsPipelines(const WorldDSLayouts &worldDSLayouts);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_SKYBOX_RENDERER_HPP
