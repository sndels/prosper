#ifndef PROSPER_DEBUG_RENDERER_HPP
#define PROSPER_DEBUG_RENDERER_HPP

#include "Camera.hpp"
#include "DebugDrawTypes.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "World.hpp"

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
        RenderResources *resources, vk::DescriptorSetLayout camDSLayout);
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
    void createDescriptorSets();
    void createGraphicsPipeline(vk::DescriptorSetLayout camDSLayout);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{};

    vk::DescriptorSetLayout _linesDSLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _linesDescriptorSets{VK_NULL_HANDLE};

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_DEBUG_RENDERER_HPP
