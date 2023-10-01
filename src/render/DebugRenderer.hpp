#ifndef PROSPER_RENDER_DEBUG_RENDERER_HPP
#define PROSPER_RENDER_DEBUG_RENDERER_HPP

#include "../gfx/Device.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../gfx/Swapchain.hpp"
#include "../scene/Camera.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "RenderResources.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class DebugRenderer
{
  public:
    enum class DrawType : uint32_t
    {
        Default = 0,
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    DebugRenderer(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout);
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer &other) = delete;
    DebugRenderer(DebugRenderer &&other) = delete;
    DebugRenderer &operator=(const DebugRenderer &other) = delete;
    DebugRenderer &operator=(DebugRenderer &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout camDSLayout);

    struct RecordInOut
    {
        ImageHandle color;
        ImageHandle depth;
    };
    void record(
        vk::CommandBuffer cb, const Camera &cam,
        const RecordInOut &inOutTargets, uint32_t nextFrame,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, const RecordInOut &inOutTargets) const;
    struct Attachments
    {
        vk::RenderingAttachmentInfo color;
        vk::RenderingAttachmentInfo depth;
    };
    [[nodiscard]] Attachments createAttachments(
        const RecordInOut &inOutTargets) const;

    void destroyGraphicsPipeline();

    void createBuffers();
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void createGraphicsPipeline(vk::DescriptorSetLayout camDSLayout);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{};
    wheels::Optional<ShaderReflection> _vertReflection;

    vk::DescriptorSetLayout _linesDSLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _linesDescriptorSets{VK_NULL_HANDLE};

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_DEBUG_RENDERER_HPP
