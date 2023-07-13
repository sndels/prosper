#ifndef PROSPER_DEPTH_OF_FIELD_GATHER_HPP
#define PROSPER_DEPTH_OF_FIELD_GATHER_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldGather
{
  public:
    enum GatherType : uint32_t
    {
        GatherType_Foreground = 0,
        GatherType_Background = 1,
        GatherType_Count,
    };

    DepthOfFieldGather(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldGather();

    DepthOfFieldGather(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather(DepthOfFieldGather &&other) = delete;
    DepthOfFieldGather &operator=(const DepthOfFieldGather &other) = delete;
    DepthOfFieldGather &operator=(DepthOfFieldGather &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Input
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCoC;
        ImageHandle dilatedTileMinMaxCoC;
    };
    struct Output
    {
        ImageHandle halfResBokehColorWeight;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Input &input, GatherType gatherType,
        uint32_t nextFrame, Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, const Input &input, const Output &output) const;

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        vk::DescriptorSet descriptorSet, GatherType gatherType,
        const Input &input, const Output &output);
    void createPipeline();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::ShaderModule, GatherType_Count> _shaderModules{{}};
    wheels::StaticArray<ShaderReflection, GatherType_Count> _shaderReflections;

    vk::DescriptorSetLayout _descriptorSetLayout;
    // Typedef so we can use capacity() as a template arg
    using DescriptorSets = wheels::StaticArray<
        vk::DescriptorSet, GatherType_Count * MAX_FRAMES_IN_FLIGHT>;
    DescriptorSets _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, GatherType_Count> _pipelines;

    uint32_t _frameIndex{0};
};

#endif // PROSPER_DEPTH_OF_FIELD_GATHER_HPP
