#ifndef PROSPER_RENDER_RT_DIRECT_ILLUMINATION_HPP
#define PROSPER_RENDER_RT_DIRECT_ILLUMINATION_HPP

#include "../gfx/Device.hpp"
#include "../scene/Camera.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "GBufferRenderer.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class RtDirectIllumination
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    RtDirectIllumination(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);
    ~RtDirectIllumination();

    RtDirectIllumination(const RtDirectIllumination &other) = delete;
    RtDirectIllumination(RtDirectIllumination &&other) = delete;
    RtDirectIllumination &operator=(const RtDirectIllumination &other) = delete;
    RtDirectIllumination &operator=(RtDirectIllumination &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        vk::DescriptorSetLayout camDSLayout,
        const World::DSLayouts &worldDSLayouts);

    void drawUi();

    struct Output
    {
        ImageHandle illumination;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const World &world, const Camera &cam,
        const GBufferRenderer::Output &gbuffer, uint32_t nextFrame,
        Profiler *profiler);

  private:
    void destroyShaders();
    void destroyPipeline();

    [[nodiscard]] bool compileShaders(
        wheels::ScopedScratch scopeAlloc,
        const World::DSLayouts &worldDSLayouts);

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        uint32_t nextFrame, const GBufferRenderer::Output &gbuffer,
        Output output);
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
    uint32_t _frameIndex{0};
};

#endif // PROSPER_RENDER_RT_DIRECT_ILLUMINATION_HPP
