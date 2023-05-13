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
        RenderResources *resources, const vk::Extent2D &renderExtent);
    ~ToneMap();

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    void recreate(const vk::Extent2D &renderExtent);

    void drawUi();

    void record(
        vk::CommandBuffer cb, uint32_t nextImage, Profiler *profiler) const;

  private:
    bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroyViewportRelated();
    void destroyPipelines();

    void createOutputImage(const vk::Extent2D &renderExtent);
    void createDescriptorSets();
    void updateDescriptorSets();
    void createPipelines();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT>
        _descriptorSets{{}};
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::Framebuffer _fbo;

    float _exposure{1.f};
};

#endif // PROSPER_TONE_MAP_HPP
