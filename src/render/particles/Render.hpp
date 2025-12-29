#ifndef PROSPER_RENDER_PARTICLES_RENDER_HPP
#define PROSPER_RENDER_PARTICLES_RENDER_HPP

#include "gfx/ShaderReflection.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"
#include "utils/Utils.hpp"
#include "vulkan/vulkan.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::particles
{

class Render
{
  public:
    Render() noexcept = default;
    ~Render();

    Render(const Render &other) = delete;
    Render(Render &&other) = delete;
    Render &operator=(const Render &other) = delete;
    Render &operator=(Render &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        vk::DescriptorSetLayout cameraDSLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout cameraDSLayout);

    struct InputOutput
    {
        BufferHandle inParticles;
        BufferHandle inIndirectArgs;
        ImageHandle inOutIllumination;
        ImageHandle inOutDepth;
    };
    void record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::Camera &cam, const InputOutput &inOut, uint32_t nextFrame);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);

    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSet ds,
        const InputOutput &inOut) const;

    void destroyGraphicsPipelines();
    void createGraphicsPipelines(vk::DescriptorSetLayout cameraDSLayout);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> m_shaderStages;
    wheels::Optional<gfx::ShaderReflection> m_vertReflection;
    wheels::Optional<gfx::ShaderReflection> m_fragReflection;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;

    vk::DescriptorSetLayout m_setLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        m_descriptorSets{VK_NULL_HANDLE};
};

} // namespace render::particles

#endif // PROSPER_RENDER_PARTICLES_RENDER_HPP
