#ifndef PROSPER_TONE_MAP_HPP
#define PROSPER_TONE_MAP_HPP

#include <functional>

#include "Device.hpp"
#include "Profiler.hpp"
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
    ToneMap(ToneMap &&other) = delete;
    ToneMap &operator=(const ToneMap &other) = delete;
    ToneMap &operator=(ToneMap &&other) = delete;

    void recompileShaders();

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig);

    [[nodiscard]] vk::CommandBuffer execute(
        uint32_t nextImage, Profiler *profiler) const;

  private:
    bool compileShaders();

    void destroySwapchainRelated();
    void destroyPipelines();

    void createOutputImage(const SwapchainConfig &swapConfig);
    void createDescriptorSet(const SwapchainConfig &swapConfig);
    void createPipelines();
    void createCommandBuffers(const SwapchainConfig &swapConfig);

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};

    vk::ShaderModule _compSM;

    vk::DescriptorSetLayout _descriptorSetLayout;
    std::vector<vk::DescriptorSet> _descriptorSets;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::vector<vk::CommandBuffer> _commandBuffers;

    vk::Framebuffer _fbo;
};

#endif // PROSPER_TONE_MAP_HPP
