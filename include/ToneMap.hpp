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
        RenderResources *resources, const SwapchainConfig &swapConfig);
    ~ToneMap();

    ToneMap(const ToneMap &other) = delete;
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void recompileShaders(wheels::ScopedScratch scopeAlloc);

    void recreate(const SwapchainConfig &swapConfig);

    void record(
        vk::CommandBuffer cb, uint32_t nextImage, Profiler *profiler) const;

  private:
    bool compileShaders(wheels::ScopedScratch scopeAlloc);

    void destroySwapchainRelated();
    void destroyPipelines();

    void createOutputImage(const SwapchainConfig &swapConfig);
    void createDescriptorSet(const SwapchainConfig &swapConfig);
    void createPipelines();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    wheels::StaticArray<vk::DescriptorSet, MAX_SWAPCHAIN_IMAGES>
        _descriptorSets;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::Framebuffer _fbo;
};

#endif // PROSPER_TONE_MAP_HPP
