#ifndef PROSPER_RENDER_DEBUG_RENDERER_HPP
#define PROSPER_RENDER_DEBUG_RENDERER_HPP

#include "gfx/Fwd.hpp"
#include "gfx/ShaderReflection.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"
#include "utils/Utils.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/static_array.hpp>

class DebugRenderer
{
  public:
    DebugRenderer() noexcept = default;
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer &other) = delete;
    DebugRenderer(DebugRenderer &&other) = delete;
    DebugRenderer &operator=(const DebugRenderer &other) = delete;
    DebugRenderer &operator=(DebugRenderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout);

    struct RecordInOut
    {
        ImageHandle color;
        ImageHandle depth;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const Camera &cam, const RecordInOut &inOutTargets,
        uint32_t nextFrame) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyGraphicsPipeline();

    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);
    void createGraphicsPipeline(vk::DescriptorSetLayout camDSLayout);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<ShaderReflection> m_vertReflection;
    wheels::Optional<ShaderReflection> m_fragReflection;

    vk::DescriptorSetLayout m_linesDSLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        m_linesDescriptorSets;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;
};

#endif // PROSPER_RENDER_DEBUG_RENDERER_HPP
