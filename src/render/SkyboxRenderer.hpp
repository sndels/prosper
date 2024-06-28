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
    SkyboxRenderer() noexcept = default;
    ~SkyboxRenderer();

    SkyboxRenderer(const SkyboxRenderer &other) = delete;
    SkyboxRenderer(SkyboxRenderer &&other) = delete;
    SkyboxRenderer &operator=(const SkyboxRenderer &other) = delete;
    SkyboxRenderer &operator=(SkyboxRenderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        const vk::DescriptorSetLayout camDSLayout,
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
        const RecordInOut &inOutTargets) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyGraphicsPipelines();

    void createGraphicsPipelines(
        const vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<ShaderReflection> m_vertReflection;
    wheels::Optional<ShaderReflection> m_fragReflection;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;
};

#endif // PROSPER_RENDER_SKYBOX_RENDERER_HPP
