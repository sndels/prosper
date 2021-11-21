#ifndef PROSPER_TONE_MAP_HPP
#define PROSPER_TONE_MAP_HPP

#include <functional>

#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"

class ToneMap
{
  public:
    ToneMap(
        Device *device, RenderResources *resources,
        const SwapchainConfig &swapConfig);
    ~ToneMap();

    ToneMap(const ToneMap &other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig);

    vk::CommandBuffer execute(const uint32_t nextImage) const;

  private:
    void destroySwapchainRelated();
    void createOutputImage(const SwapchainConfig &swapConfig);
    void createDescriptorSet(const SwapchainConfig &swapConfig);
    void createPipelines(const SwapchainConfig &swapConfig);
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device = nullptr;
    RenderResources *_resources = nullptr;

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
};

#endif // PROSPER_TONE_MAP_HPP