#ifndef PROSPER_RENDER_DEBUG_RENDERER_HPP
#define PROSPER_RENDER_DEBUG_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../gfx/ShaderReflection.hpp"
#include "../scene/DebugDrawTypes.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/static_array.hpp>

class DebugRenderer
{
  public:
    enum class DrawType : uint32_t
    {
        DEBUG_DRAW_TYPES_AND_COUNT
    };

    DebugRenderer() noexcept = default;
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer &other) = delete;
    DebugRenderer(DebugRenderer &&other) = delete;
    DebugRenderer &operator=(const DebugRenderer &other) = delete;
    DebugRenderer &operator=(DebugRenderer &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout camDSLayout);

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
        const Camera &cam, const RecordInOut &inOutTargets, uint32_t nextFrame,
        Profiler *profiler) const;

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

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

    bool _initialized{false};
    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::PipelineShaderStageCreateInfo, 2> _shaderStages{};
    wheels::Optional<ShaderReflection> _vertReflection;
    wheels::Optional<ShaderReflection> _fragReflection;

    vk::DescriptorSetLayout _linesDSLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _linesDescriptorSets;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_RENDER_DEBUG_RENDERER_HPP
