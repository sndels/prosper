#ifndef PROSPER_DOWN_SAMPLER_HPP
#define PROSPER_DOWN_SAMPLER_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"

class DownSampler
{
  public:
    static const size_t sMaxMips = 12;

    // NOTE:
    // These are currently tied to binding sets so they need to be unique across
    // the frame. If e.g. two different depth downsamples are needed, two
    // different operations should be defined.
    // TODO:
    // Once multiple copies of the same operation are needed, add support for
    // them without duplicating shaders, pipelines etc.
    enum Operation : uint32_t
    {
        First = 0,
        MaxDepth = 0,
        Count = 1,
    };

    DownSampler(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~DownSampler();

    DownSampler(const DownSampler &other) = delete;
    DownSampler(DownSampler &&other) = delete;
    DownSampler &operator=(const DownSampler &other) = delete;
    DownSampler &operator=(DownSampler &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    struct Output
    {
        ImageHandle downSampled;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle input, Operation operation,
        uint32_t nextFrame, Profiler *profiler);

  private:
    bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyPipelines();

    void createResources(
        const vk::Extent2D &size, Operation op, Output &output,
        BufferHandle &globalCounter);
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc,
        DescriptorAllocator *staticDescriptorsAlloc);
    void createPipelines();
    void createSamplers();

    struct BoundResources
    {
        ImageHandle input;
        ImageHandle output;
        BufferHandle globalCounter;
    };
    void updateDescriptorSet(
        vk::DescriptorSet descriptorSet, const BoundResources &resources,
        Operation operation);

    void recordBarriers(
        vk::CommandBuffer cb, const BoundResources &resources) const;

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    wheels::StaticArray<vk::ShaderModule, Operation::Count> _compSMs;
    wheels::StaticArray<ShaderReflection, Operation::Count> _shaderReflections;

    vk::DescriptorSetLayout _descriptorSetLayout;

    // Typedef so we can use capacity() as a template arg
    using DescriptorSets = wheels::StaticArray<
        vk::DescriptorSet, Operation::Count * MAX_FRAMES_IN_FLIGHT>;
    DescriptorSets _descriptorSets{{}};

    vk::PipelineLayout _pipelineLayout;
    wheels::StaticArray<vk::Pipeline, Operation::Count> _pipelines;
    wheels::StaticArray<vk::Sampler, Operation::Count> _samplers;
};

#endif // PROSPER_DOWN_SAMPLER_HPP
