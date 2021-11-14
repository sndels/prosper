
#ifndef PROSPER_IMGUI_CONTEXT_HPP
#define PROSPER_IMGUI_CONTEXT_HPP

#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"

#include <glfw/glfw3.h>

class ImGuiRenderer
{
  public:
    ImGuiRenderer(
        Device *device, RenderResources *resources, GLFWwindow *window,
        const SwapchainConfig &swapConfig);
    ~ImGuiRenderer();

    void startFrame() const;
    vk::CommandBuffer endFrame(
        const vk::Rect2D &renderArea, const uint32_t nextImage);

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig);

  private:
    void createRenderPass(const vk::Format &colorFormat);

    void destroySwapchainRelated();
    void createDescriptorPool(const SwapchainConfig &swapConfig);

  private:
    Device *_device = nullptr;
    RenderResources *_resources = nullptr;
    vk::DescriptorPool _descriptorPool;
    vk::RenderPass _renderpass;
    vk::Framebuffer _fbo;
    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_IMGUI_CONTEXT_HPP
