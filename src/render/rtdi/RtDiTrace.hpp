#ifndef PROSPER_RENDER_RTDI_TRACE_HPP
#define PROSPER_RENDER_RTDI_TRACE_HPP

#include "../../gfx/Fwd.hpp"
#include "../../gfx/Resources.hpp"
#include "../../gfx/ShaderReflection.hpp"
#include "../../scene/DebugDrawTypes.hpp"
#include "../../scene/Fwd.hpp"
#include "../../utils/Fwd.hpp"
#include "../Fwd.hpp"
#include "../RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtDiTrace
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    RtDiTrace(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);
    ~RtDiTrace();

    RtDiTrace(const RtDiTrace &other) = delete;
    RtDiTrace(RtDiTrace &&other) = delete;
    RtDiTrace &operator=(const RtDiTrace &other) = delete;
    RtDiTrace &operator=(RtDiTrace &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const WorldDSLayouts &worldDSLayouts);

    void drawUi();

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
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, const Input &input,
        bool resetAccumulation, uint32_t nextFrame, Profiler *profiler);
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

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 4> _shaderStages{{}};
    wheels::StaticArray<vk::RayTracingShaderGroupCreateInfoKHR, 3>
        _shaderGroups{{}};
    wheels::Optional<ShaderReflection> _raygenReflection;
    wheels::Optional<ShaderReflection> _rayMissReflection;
    wheels::Optional<ShaderReflection> _closestHitReflection;
    wheels::Optional<ShaderReflection> _anyHitReflection;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::DeviceSize _sbtGroupSize{0};
    Buffer _shaderBindingTable;

    DrawType _drawType{DrawType::Default};
    bool _accumulationDirty{true};
    bool _accumulate{false};
    uint32_t _frameIndex{0};

    ImageHandle _previousIllumination;
};

#endif // PROSPER_RENDER_RTDI_TRACE_HPP
