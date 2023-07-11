#ifndef PROSPER_DEPTH_OF_FIELD_DILATE_HPP
#define PROSPER_DEPTH_OF_FIELD_DILATE_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Utils.hpp"

// Based on A Life of a Bokeh by Guillaume Abadie
// https://advances.realtimerendering.com/s2018/index.htm

class DepthOfFieldDilate
{
  public:
    DepthOfFieldDilate(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);

    ~DepthOfFieldDilate();

    DepthOfFieldDilate(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate(DepthOfFieldDilate &&other) = delete;
    DepthOfFieldDilate &operator=(const DepthOfFieldDilate &other) = delete;
    DepthOfFieldDilate &operator=(DepthOfFieldDilate &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Output
    {
        ImageHandle dilatedTileMinMaxCoC;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle tileMinMaxCoC, uint32_t nextFrame,
        Profiler *profiler);

  private:
    [[nodiscard]] bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void recordBarriers(
        vk::CommandBuffer cb, ImageHandle tileMinMaxCoC,
        const Output &output) const;

    void destroyPipelines();

    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void updateDescriptorSet(
        uint32_t nextFrame, ImageHandle tileMinMaxCoC, const Output &output);
    void createPipeline();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;
    wheels::Optional<ShaderReflection> _shaderReflection;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
};

#endif // PROSPER_DEPTH_OF_FIELD_DILATE_HPP
