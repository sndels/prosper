#include "ImGuiRenderer.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>

#include <wheels/containers/static_array.hpp>

#include "../gfx/Swapchain.hpp"
#include "../gfx/VkUtils.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

using namespace wheels;

namespace
{

constexpr void checkSuccessImGui(VkResult err)
{
    checkSuccess(static_cast<vk::Result>(err), "ImGui");
}

const char *const sIniFilename = "prosper_imgui.ini";
const char *const sDefaultIniFilename = "default_prosper_imgui.ini";

// Copied from imgui.h, let's not pollute the whole project with these
ImVec4 operator+(const ImVec4 &lhs, const ImVec4 &rhs)
{
    return ImVec4{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
}
ImVec4 operator*(const ImVec4 &lhs, const ImVec4 &rhs)
{
    return ImVec4{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
}

} // namespace

ImGuiRenderer::~ImGuiRenderer()
{
    if (_device != nullptr)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        _device->logical().destroy(_descriptorPool);
    }
}

void ImGuiRenderer::init(
    Device *device, RenderResources *resources, GLFWwindow *window,
    const SwapchainConfig &swapConfig)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(resources != nullptr);
    WHEELS_ASSERT(window != nullptr);

    _device = device;
    _resources = resources;

    printf("Creating ImGuiRenderer\n");

    createDescriptorPool();

    ImGui::CreateContext();

    if (!std::filesystem::exists(sIniFilename))
    {
        printf("ImGui ini not found, copying default ini into working dir\n");
        try
        {
            std::filesystem::copy(resPath(sDefaultIniFilename), sIniFilename);
        }
        catch (...)
        {
            fprintf(
                stderr, "Failed to copy default imgui config into working "
                        "directory.\n");
        }
    }

    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = sIniFilename;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, false);
    ImGui_ImplVulkan_InitInfo init_info = {
        .Instance = _device->instance(),
        .PhysicalDevice = _device->physical(),
        .Device = _device->logical(),
        .QueueFamily = *_device->queueFamilies().graphicsFamily,
        .Queue = _device->graphicsQueue(),
        .DescriptorPool = _descriptorPool,
        .MinImageCount = swapConfig.imageCount,
        .ImageCount = swapConfig.imageCount,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineCache = vk::PipelineCache{},
        .UseDynamicRendering = true,
        .PipelineRenderingCreateInfo =
            vk::PipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &sFinalCompositeFormat,
            }, // TODO: Pass in VMA callbacks?
        .CheckVkResultFn = checkSuccessImGui,
        .MinAllocationSize =
            static_cast<VkDeviceSize>(1024) * static_cast<VkDeviceSize>(1024),
    };
    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    setStyle();

    _initialized = true;
}

// NOLINTNEXTLINE could be static, but requires an instance TODO: Singleton?
void ImGuiRenderer::startFrame(Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);
    const auto _s = profiler->createCpuScope("ImGui::startFrame");

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // The render is drawn onto the central node before ui is rendered
    const ImGuiDockNodeFlags dockFlags =
        ImGuiDockNodeFlags_NoDockingInCentralNode |
        ImGuiDockNodeFlags_PassthruCentralNode;
    _dockAreaID = ImGui::DockSpaceOverViewport(nullptr, dockFlags);
}

void ImGuiRenderer::endFrame(
    vk::CommandBuffer cb, const vk::Rect2D &renderArea, ImageHandle inOutColor,
    Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    {
        const auto _s = profiler->createCpuScope("ImGui::render");
        ImGui::Render();
    }
    ImDrawData *drawData = ImGui::GetDrawData();

    {
        _resources->images.transition(
            cb, inOutColor, ImageState::ColorAttachmentReadWrite);

        const auto _s = profiler->createCpuGpuScope(cb, "ImGui::draw", true);

        const vk::RenderingAttachmentInfo attachment{
            .imageView = _resources->images.resource(inOutColor).view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachment,
        });

        ImGui_ImplVulkan_RenderDrawData(drawData, cb);

        cb.endRendering();
    }
}

ImVec2 ImGuiRenderer::centerAreaOffset() const
{
    WHEELS_ASSERT(_initialized);

    const ImGuiDockNode *node = ImGui::DockBuilderGetCentralNode(_dockAreaID);
    WHEELS_ASSERT(node != nullptr);

    return node->Pos;
}

ImVec2 ImGuiRenderer::centerAreaSize() const
{
    WHEELS_ASSERT(_initialized);

    const ImGuiDockNode *node = ImGui::DockBuilderGetCentralNode(_dockAreaID);
    WHEELS_ASSERT(node != nullptr);

    return node->Size;
}

void ImGuiRenderer::createDescriptorPool()
{
    // One descriptor for the font. More are needed if things like textures are
    // loaded into imgui itselfe
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
    };
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        });
}

void ImGuiRenderer::setStyle()
{
    ImGuiStyle &style = ImGui::GetStyle();

    // Let's be pointy
    style.TabRounding = 0.f;
    style.ScrollbarRounding = 0.f;
    style.WindowMenuButtonPosition = ImGuiDir_None;

    ImVec4 *colors = reinterpret_cast<ImVec4 *>(style.Colors);

    // Lighter dark mode, closer to what most apps are doing these days
    const ImVec4 colorBg{0.12f, 0.12f, 0.12f, 0.90f};
    const ImVec4 colorBgLight{0.16f, 0.16f, 0.16f, 0.90f};
    const ImVec4 colorTransparent{0.00f, 0.00f, 0.00f, 0.00f};
    const ImVec4 colorItemDark{0.09f, 0.09f, 0.09f, 0.90f};
    const ImVec4 colorItemDelta{0.12f, 0.12f, 0.12f, 0.90f};
    const ImVec4 colorItem = colorItemDark + colorItemDelta;
    const ImVec4 colorItemHighlight = colorItem + colorItemDelta;
    const ImVec4 colorItemBrightHighlight = colorItemHighlight + colorItemDelta;
    const ImVec4 colorAccent = ImVec4{0.13f, 0.33f, 0.58f, 1.00f};
    const ImVec4 colorAccentDark =
        colorAccent * ImVec4{0.85f, 0.85f, 0.85f, 1.f};
    const ImVec4 colorAccentBright =
        colorAccent * ImVec4{1.15f, 1.15f, 1.15f, 1.f};

    colors[ImGuiCol_Text] = ImVec4{0.90f, 0.90f, 0.90f, 1.00f};
    colors[ImGuiCol_TextDisabled] = ImVec4{0.50f, 0.50f, 0.50f, 1.00f};
    colors[ImGuiCol_WindowBg] = colorBg;
    colors[ImGuiCol_ChildBg] = colorTransparent;
    colors[ImGuiCol_PopupBg] = colorBg;
    colors[ImGuiCol_Border] = ImVec4{0.43f, 0.43f, 0.43f, 0.50f};
    colors[ImGuiCol_BorderShadow] = colorTransparent;
    colors[ImGuiCol_FrameBg] = colorItemDark;
    colors[ImGuiCol_FrameBgHovered] = colorItem;
    colors[ImGuiCol_FrameBgActive] = colorItemHighlight;
    colors[ImGuiCol_TitleBg] = colorItem;
    colors[ImGuiCol_TitleBgActive] = colorItem;
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg] = colorBgLight;
    colors[ImGuiCol_ScrollbarBg] = colorItemDark;
    colors[ImGuiCol_ScrollbarGrab] = colorItem;
    colors[ImGuiCol_ScrollbarGrabHovered] = colorItemHighlight;
    colors[ImGuiCol_ScrollbarGrabActive] = colorItem;
    colors[ImGuiCol_CheckMark] = colorAccentBright;
    colors[ImGuiCol_SliderGrab] = colorItemHighlight;
    colors[ImGuiCol_SliderGrabActive] = colorItemBrightHighlight;
    colors[ImGuiCol_Button] = colorAccentDark;
    colors[ImGuiCol_ButtonHovered] = colorAccent;
    colors[ImGuiCol_ButtonActive] = colorAccentBright;
    colors[ImGuiCol_Header] = colorItem;
    colors[ImGuiCol_HeaderHovered] = colorItemHighlight;
    colors[ImGuiCol_HeaderActive] = colorItemBrightHighlight;
    colors[ImGuiCol_Separator] = colorItem;
    colors[ImGuiCol_SeparatorHovered] = colorItemHighlight;
    colors[ImGuiCol_SeparatorActive] = colorItem;
    colors[ImGuiCol_ResizeGrip] = colorItem;
    colors[ImGuiCol_ResizeGripHovered] = colorItemHighlight;
    colors[ImGuiCol_ResizeGripActive] = colorItemBrightHighlight;
    colors[ImGuiCol_Tab] = colorItem;
    colors[ImGuiCol_TabHovered] = colorItemHighlight;
    colors[ImGuiCol_TabActive] = colorItemHighlight;
    colors[ImGuiCol_TabUnfocused] = colorItem;
    colors[ImGuiCol_TabUnfocusedActive] = colorItemHighlight;
    colors[ImGuiCol_DockingPreview] =
        colors[ImGuiCol_HeaderHovered] * ImVec4(1.0f, 1.0f, 1.0f, 0.7f);
    colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
    colors[ImGuiCol_TableBorderStrong] =
        ImVec4(0.31f, 0.31f, 0.35f, 1.00f); // Prefer using Alpha=1.0 here
    colors[ImGuiCol_TableBorderLight] =
        ImVec4(0.23f, 0.23f, 0.25f, 1.00f); // Prefer using Alpha=1.0 here
    colors[ImGuiCol_TableRowBg] = colorTransparent;
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg] = colorItemBrightHighlight;
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = colorItemBrightHighlight;
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}
