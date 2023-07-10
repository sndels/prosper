#ifndef PROSPER_DEPTH_OF_FIELD_SETUP_HPP
#define PROSPER_DEPTH_OF_FIELD_SETUP_HPP

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

class DepthOfFieldSetup
{
  public:
    DepthOfFieldSetup(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources, DescriptorAllocator *staticDescriptorsAlloc,
        vk::DescriptorSetLayout cameraDsLayout);

    ~DepthOfFieldSetup();

    DepthOfFieldSetup(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup(DepthOfFieldSetup &&other) = delete;
    DepthOfFieldSetup &operator=(const DepthOfFieldSetup &other) = delete;
    DepthOfFieldSetup &operator=(DepthOfFieldSetup &&other) = delete;

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        vk::DescriptorSetLayout cameraDsLayout);

    struct Input
    {
        ImageHandle illumination;
        ImageHandle depth;
    };
    struct Output
    {
        ImageHandle halfResIllumination;
        ImageHandle halfResCircleOfConfusion;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, const Camera &cam, const Input &input,
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
        uint32_t nextFrame, const Input &input, const Output &output);
    void createPipeline(vk::DescriptorSetLayout cameraDsLayout);

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

#endif // PROSPER_DEPTH_OF_FIELD_SETUP_HPP
