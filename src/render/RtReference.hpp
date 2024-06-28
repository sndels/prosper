#ifndef PROSPER_RENDER_RT_REFERENCE_HPP
#define PROSPER_RENDER_RT_REFERENCE_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DrawType.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtReference
{
  public:
    static constexpr uint32_t sMaxBounces = 6;

    RtReference() noexcept = default;
    ~RtReference();

    RtReference(const RtReference &other) = delete;
    RtReference(RtReference &&other) = delete;
    RtReference &operator=(const RtReference &other) = delete;
    RtReference &operator=(RtReference &&other) = delete;

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

    void drawUi();

    struct Options
    {
        bool depthOfField{false};
        bool ibl{false};
        bool colorDirty{false};
        DrawType drawType{DrawType::Default};
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
        const Camera &cam, const vk::Rect2D &renderArea, const Options &options,
        uint32_t nextFrame, Profiler *profiler);
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
        ImageHandle illumination);
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
    bool m_accumulate{true};
    bool m_clampIndirect{true};
    uint32_t m_frameIndex{0};
    uint32_t m_rouletteStartBounce{3};
    uint32_t m_maxBounces{sMaxBounces};

    ImageHandle m_previousIllumination;
};

#endif // PROSPER_RENDER_RT_REFERENCE_HPP
