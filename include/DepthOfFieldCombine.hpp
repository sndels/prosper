#ifndef PROSPER_DEPTH_OF_FIELD_COMBINE_HPP
#define PROSPER_DEPTH_OF_FIELD_COMBINE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "Camera.hpp"
#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldCombine
{
  public:
    DepthOfFieldCombine(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldCombine();

    DepthOfFieldCombine(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine(DepthOfFieldCombine &&other) = delete;
    DepthOfFieldCombine &operator=(const DepthOfFieldCombine &other) = delete;
    DepthOfFieldCombine &operator=(DepthOfFieldCombine &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Input
    {
        ImageHandle halfResFgBokehWeight;
        ImageHandle halfResBgBokehWeight;
        ImageHandle halfResCircleOfConfusion;
        ImageHandle illumination;
    };
    struct Output
    {
        ImageHandle combinedIlluminationDoF;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Input &input, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, const Input &input, const Output &output) const;

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        uint32_t nextFrame, const Input &input, const Output &output);
    void createPipeline();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _shaderModule;
    wheels::Optional<ShaderReflection> _shaderReflection;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_DEPTH_OF_FIELD_COMBINE_HPP
