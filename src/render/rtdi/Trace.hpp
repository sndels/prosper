#ifndef PROSPER_RENDER_RTDI_TRACE_HPP
#define PROSPER_RENDER_RTDI_TRACE_HPP

#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "gfx/ShaderReflection.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/DrawType.hpp"
#include "scene/Fwd.hpp"
#include "utils/Utils.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

namespace render::rtdi
{

class Trace
{
  public:
    Trace() noexcept = default;
    ~Trace();

    Trace(const Trace &other) = delete;
    Trace(Trace &&other) = delete;
    Trace &operator=(const Trace &other) = delete;
    Trace &operator=(Trace &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);

    struct Input
    {
        const GBuffer &gbuffer;
        ImageHandle reservoirs;
    };
    struct Output
    {
        ImageHandle diffuseIllumination;
        ImageHandle specularIllumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        scene::World &world, const scene::Camera &cam, const Input &input,
        bool resetAccumulation, scene::DrawType drawType, uint32_t nextFrame);
    void releasePreserved();

  private:
    void destroyShaders();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const scene::WorldDSLayouts &worldDSLayouts);

    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);
    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame,
        const Input &inputs, const Output &output);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const scene::WorldDSLayouts &worldDSLayouts);
    void createShaderBindingTable(wheels::ScopedScratch scopeAlloc);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 4> m_shaderStages;
    wheels::StaticArray<vk::RayTracingShaderGroupCreateInfoKHR, 3>
        m_shaderGroups;
    wheels::Optional<gfx::ShaderReflection> m_raygenReflection;
    wheels::Optional<gfx::ShaderReflection> m_rayMissReflection;
    wheels::Optional<gfx::ShaderReflection> m_closestHitReflection;
    wheels::Optional<gfx::ShaderReflection> m_anyHitReflection;

    vk::DescriptorSetLayout m_descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        m_descriptorSets;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;

    vk::DeviceSize m_sbtGroupSize{0};
    gfx::Buffer m_shaderBindingTable;

    bool m_accumulationDirty{true};
    bool m_accumulate{false};
    uint32_t m_frameIndex{0};

    ImageHandle m_previousDiffuseIllumination;
    ImageHandle m_previousSpecularIllumination;
};

} // namespace render::rtdi

#endif // PROSPER_RENDER_RTDI_TRACE_HPP
