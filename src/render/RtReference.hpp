#ifndef PROSPER_RENDER_RT_REFERENCE_HPP
#define PROSPER_RENDER_RT_REFERENCE_HPP

#include "../gfx/Device.hpp"
#include "../scene/Camera.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtReference
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    static constexpr uint32_t sMaxBounces = 6;

    RtReference(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~RtReference();

    RtReference(const RtReference &other) = delete;
    RtReference(RtReference &&other) = delete;
    RtReference &operator=(const RtReference &other) = delete;
    RtReference &operator=(RtReference &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    struct Options
    {
        bool depthOfField{false};
        bool ibl{false};
        bool colorDirty{false};
    };
    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const vk::Rect2D &renderArea, const Options &options,
        uint32_t nextFrame, Profiler *profiler);
    void releasePreserved();

  private:
    void destroyShaders();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(uint32_t nextFrame, ImageHandle illumination);
    void createPipeline(
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
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
    bool _accumulate{true};
    bool _clampIndirect{true};
    uint32_t _frameIndex{0};
    uint32_t _rouletteStartBounce{3};
    uint32_t _maxBounces{sMaxBounces};

    ImageHandle _previousIllumination;
};

#endif // PROSPER_RENDER_RT_REFERENCE_HPP
