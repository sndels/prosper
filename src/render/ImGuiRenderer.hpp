
#ifndef PROSPER_RENDER_IMGUI_RENDERER_HPP
#define PROSPER_RENDER_IMGUI_RENDERER_HPP

#include "gfx/Fwd.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"

#include <imgui.h>
#include <vulkan/vulkan.hpp>

class ImGuiRenderer
{
  public:
    ImGuiRenderer() noexcept = default;
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer &other) = delete;
    ImGuiRenderer(ImGuiRenderer &&other) = delete;
    ImGuiRenderer &operator=(const ImGuiRenderer &other) = delete;
    ImGuiRenderer &operator=(ImGuiRenderer &&other) = delete;

    void init(const SwapchainConfig &swapConfig);

    void startFrame();
    void endFrame(
        vk::CommandBuffer cb, const vk::Rect2D &renderArea,
        ImageHandle inOutColor) const;

    [[nodiscard]] ImVec2 centerAreaOffset() const;
    [[nodiscard]] ImVec2 centerAreaSize() const;

  private:
    void createDescriptorPool();

    static void setStyle();

    bool m_initialized{false};
    vk::DescriptorPool m_descriptorPool;
    ImGuiID m_dockAreaID{0};
};

#endif // PROSPER_RENDER_IMGUI_RENDERER_HPP
