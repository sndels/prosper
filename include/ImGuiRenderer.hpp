
#ifndef PROSPER_IMGUI_CONTEXT_HPP
#define PROSPER_IMGUI_CONTEXT_HPP

#include "Device.hpp"
#include "Swapchain.hpp"

#include <glfw/glfw3.h>

class ImGuiRenderer
{
  public:
    ImGuiRenderer(
        std::shared_ptr<Device> device, GLFWwindow *window,
        const SwapchainConfig &swapConfig);
    ~ImGuiRenderer();

    void startFrame() const;
    vk::CommandBuffer endFrame(
        const Image &outputImage, const uint32_t nextImage);

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig);

  private:
    void createRenderPass(const vk::Format &colorFormat);

    void destroySwapchainRelated();
    void createDescriptorPool(const SwapchainConfig &swapConfig);
    void recreateFramebuffer(const Image &image);

  private:
    std::shared_ptr<Device> _device;
    vk::DescriptorPool _descriptorPool;
    vk::RenderPass _renderpass;
    std::pair<vk::Image, vk::Framebuffer> _fbo;
    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_IMGUI_CONTEXT_HPP
