
#ifndef PROSPER_IMGUI_RENDERER_HPP
#define PROSPER_IMGUI_RENDERER_HPP

#include "Device.hpp"
#include "RenderResources.hpp"
#include "Swapchain.hpp"

#include <GLFW/glfw3.h>

class ImGuiRenderer
{
  public:
    ImGuiRenderer(
        Device *device, RenderResources *resources, GLFWwindow *window,
        const SwapchainConfig &swapConfig);
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer &other) = delete;
    ImGuiRenderer(ImGuiRenderer &&other) = delete;
    ImGuiRenderer &operator=(const ImGuiRenderer &other) = delete;
    ImGuiRenderer &operator=(ImGuiRenderer &&other) = delete;

    void startFrame() const;
    [[nodiscard]] vk::CommandBuffer endFrame(
        const vk::Rect2D &renderArea, uint32_t nextImage);

    void recreateSwapchainRelated(const SwapchainConfig &swapConfig);

  private:
    void createRenderPass(const vk::Format &colorFormat);

    void destroySwapchainRelated();
    void createDescriptorPool();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};
    vk::DescriptorPool _descriptorPool;
    vk::RenderPass _renderpass;
    vk::Framebuffer _fbo;
    std::vector<vk::CommandBuffer> _commandBuffers;
};

#endif // PROSPER_IMGUI_RENDERER_HPP
