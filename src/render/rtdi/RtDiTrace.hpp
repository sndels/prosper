#ifndef PROSPER_RENDER_RTDI_TRACE_HPP
#define PROSPER_RENDER_RTDI_TRACE_HPP

#include "../../gfx/Fwd.hpp"
#include "../../gfx/Resources.hpp"
#include "../../gfx/ShaderReflection.hpp"
#include "../../scene/DrawType.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtDiTrace
{
  public:
    RtDiTrace() noexcept = default;
    ~RtDiTrace();

    RtDiTrace(const RtDiTrace &other) = delete;
    RtDiTrace(RtDiTrace &&other) = delete;
    RtDiTrace &operator=(const RtDiTrace &other) = delete;
    RtDiTrace &operator=(RtDiTrace &&other) = delete;

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

    struct Input
    {
        const GBufferRendererOutput &gbuffer;
        ImageHandle reservoirs;
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
        const Camera &cam, const Input &input, bool resetAccumulation,
        DrawType drawType, uint32_t nextFrame, Profiler *profiler);
    void releasePreserved();

  private:
    void destroyShaders();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDSLayouts);

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        wheels::ScopedScratch scopeAlloc, uint32_t nextFrame,
        Input const &inputs, ImageHandle illumination);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);
    void createShaderBindingTable(wheels::ScopedScratch scopeAlloc);

    bool m_initialized{false};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 4> m_shaderStages;
    wheels::StaticArray<vk::RayTracingShaderGroupCreateInfoKHR, 3>
        m_shaderGroups;
    wheels::Optional<ShaderReflection> m_raygenReflection;
    wheels::Optional<ShaderReflection> m_rayMissReflection;
    wheels::Optional<ShaderReflection> m_closestHitReflection;
    wheels::Optional<ShaderReflection> m_anyHitReflection;

    vk::DescriptorSetLayout m_descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        m_descriptorSets;

    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline m_pipeline;

    vk::DeviceSize m_sbtGroupSize{0};
    Buffer m_shaderBindingTable;

    bool m_accumulationDirty{true};
    bool m_accumulate{false};
    uint32_t m_frameIndex{0};

    ImageHandle m_previousIllumination;
};

#endif // PROSPER_RENDER_RTDI_TRACE_HPP
