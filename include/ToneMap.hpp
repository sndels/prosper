#ifndef PROSPER_TONE_MAP_HPP
#define PROSPER_TONE_MAP_HPP

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

#include "Device.hpp"
#include "Profiler.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"
#include "Utils.hpp"

class ToneMap
{
  public:
    ToneMap(
        wheels::ScopedScratch scopeAlloc, Device *device,
        RenderResources *resources,
        DescriptorAllocator *staticDescriptorsAlloc);
    ~ToneMap();

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    void drawUi();

    struct Output
    {
        ImageHandle toneMapped;
    };
    [[nodiscard]] Output record(
        vk::CommandBuffer cb, ImageHandle inColor, uint32_t nextFrame,
        Profiler *profiler);

  private:
    bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyPipelines();

    void createOutputImage(const vk::Extent2D &renderExtent);
    void createDescriptorSets(DescriptorAllocator *staticDescriptorsAlloc);
    void createPipelines();

    Output createOutputs(const vk::Extent2D &size);

    struct BoundImages
    {
        ImageHandle inColor;
        ImageHandle toneMapped;
    };
    void updateDescriptorSet(uint32_t nextFrame, const BoundImages &images);

    void recordBarriers(vk::CommandBuffer cb, const BoundImages &images) const;

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::Extent2D _extent{};

    float _exposure{1.f};
};

#endif // PROSPER_TONE_MAP_HPP
