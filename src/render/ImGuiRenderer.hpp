
#ifndef PROSPER_RENDER_IMGUI_RENDERER_HPP
#define PROSPER_RENDER_IMGUI_RENDERER_HPP

#include "../gfx/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "Fwd.hpp"
#include <imgui.h>
#include <vulkan/vulkan.hpp>

extern "C"
{
    // Let's assume GLFW is stable enough that a fwd decl is not a problem
    // The header is pretty thick
    struct GLFWwindow;
}

class ImGuiRenderer
{
  public:
    ImGuiRenderer(
        Device *device, RenderResources *resources,
        const vk::Extent2D &renderExtent, GLFWwindow *window,
        const SwapchainConfig &swapConfig);
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer &other) = delete;
    ImGuiRenderer(ImGuiRenderer &&other) = delete;
    ImGuiRenderer &operator=(const ImGuiRenderer &other) = delete;
    ImGuiRenderer &operator=(ImGuiRenderer &&other) = delete;

    void startFrame(Profiler *profiler);
    void endFrame(
        vk::CommandBuffer cb, const vk::Rect2D &renderArea, Profiler *profiler);

    [[nodiscard]] ImVec2 centerAreaOffset() const;
    [[nodiscard]] ImVec2 centerAreaSize() const;

    void recreate(const vk::Extent2D &renderExtent);

  private:
    void createRenderPass();

    void destroySwapchainRelated();
    void createDescriptorPool();

    Device *_device{nullptr};
    RenderResources *_resources{nullptr};
    vk::DescriptorPool _descriptorPool;
    vk::RenderPass _renderpass;
    vk::Framebuffer _fbo;
    ImGuiID _dockAreaID{0};
};

#endif // PROSPER_RENDER_IMGUI_RENDERER_HPP
