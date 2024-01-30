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
    SkyboxRenderer() = default;
    ~SkyboxRenderer();

    SkyboxRenderer(const SkyboxRenderer &other) = delete;
    SkyboxRenderer(SkyboxRenderer &&other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(SkyboxRenderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, const vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    struct RecordInOut
    {
        ImageHandle illumination;
        ImageHandle velocity;
        ImageHandle depth;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &camera,
        const RecordInOut &inOutTargets, Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    struct Attachments
    {
        wheels::StaticArray<vk::RenderingAttachmentInfo, 2> color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets) const;

    void destroyGraphicsPipelines();

    void createGraphicsPipelines(
        const vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    bool _initialized{false};
    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages;
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_SKYBOX_RENDERER_HPP
