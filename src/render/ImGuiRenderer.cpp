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

using namespace wheels;

namespace
{

constexpr void checkSuccessImGui(VkResult err)
{
    checkSuccess(static_cast<vk::Result>(err), "ImGui");
}

const vk::Format sFinalCompositeFormat = vk::Format::eR8G8B8A8Unorm;

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
    destroySwapchainRelated();

    if (_device != nullptr)
    {
        _device->logical().destroy(_renderpass);
        _device->logical().destroy(_descriptorPool);

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
}

void ImGuiRenderer::init(
    Device *device, RenderResources *resources,
    const vk::Extent2D &renderExtent, GLFWwindow *window,
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
    createRenderPass();

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
        .PipelineCache = vk::PipelineCache{},
        .DescriptorPool = _descriptorPool,
        .MinImageCount = swapConfig.imageCount,
        .ImageCount = swapConfig.imageCount,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        // TODO: Pass in VMA callbacks?
        .CheckVkResultFn = checkSuccessImGui,
    };
    ImGui_ImplVulkan_Init(&init_info, _renderpass);

    recreate(renderExtent);

    auto buffer = _device->beginGraphicsCommands();

    ImGui_ImplVulkan_CreateFontsTexture(static_cast<VkCommandBuffer>(buffer));

    _device->endGraphicsCommands(buffer);

    ImGui_ImplVulkan_DestroyFontUploadObjects();

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
    vk::CommandBuffer cb, const vk::Rect2D &renderArea, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(profiler != nullptr);

    {
        const auto _s = profiler->createCpuScope("ImGui::render");
        ImGui::Render();
    }
    ImDrawData *drawData = ImGui::GetDrawData();

    {
        _resources->finalComposite.transition(
            cb, ImageState::ColorAttachmentReadWrite);

        const auto _s = profiler->createCpuGpuScope(cb, "ImGui::draw", true);

        cb.beginRenderPass(
            vk::RenderPassBeginInfo{
                .renderPass = _renderpass,
                .framebuffer = _fbo,
                .renderArea = renderArea,
            },
            vk::SubpassContents::eInline);

        ImGui_ImplVulkan_RenderDrawData(drawData, cb);

        cb.endRenderPass();
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

void ImGuiRenderer::createRenderPass()
{
    const vk::AttachmentDescription attachment = {
        .format = sFinalCompositeFormat,
        .samples = vk::SampleCountFlagBits::e1,
        // Assume this works on a populated target
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    const vk::AttachmentReference color_attachment = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    };
    _renderpass = _device->logical().createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
    });

    _device->logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eRenderPass,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkRenderPass>(_renderpass)),
            .pObjectName = "ImGui",
        });
}

void ImGuiRenderer::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        _device->logical().destroy(_fbo);
        _device->destroy(_resources->finalComposite);
    }
}

void ImGuiRenderer::recreate(const vk::Extent2D &renderExtent)
{
    WHEELS_ASSERT(_device != nullptr);

    destroySwapchainRelated();

    auto &image = _resources->finalComposite;
    image = _device->createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = sFinalCompositeFormat,
                .width = renderExtent.width,
                .height = renderExtent.height,
                .usageFlags =
                    vk::ImageUsageFlagBits::eColorAttachment | // Render
                    vk::ImageUsageFlagBits::eTransferDst |     // Blit from tone
                                                               // mapped
                    vk::ImageUsageFlagBits::eTransferSrc, // Blit to swap image
            },
        .debugName = "ui",
    });

    _fbo = _device->logical().createFramebuffer(vk::FramebufferCreateInfo{
        .renderPass = _renderpass,
        .attachmentCount = 1,
        .pAttachments = &image.view,
        .width = image.extent.width,
        .height = image.extent.height,
        .layers = 1,
    });
}

void ImGuiRenderer::createDescriptorPool()
{
    const uint32_t maxSets = 1000;
    const StaticArray poolSizes{{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eSampler,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eSampledImage,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageImage,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformTexelBuffer,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageTexelBuffer,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBufferDynamic,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageBufferDynamic,
            .descriptorCount = maxSets,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eInputAttachment,
            .descriptorCount = maxSets,
        },
    }};
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets * asserted_cast<uint32_t>(poolSizes.size()),
            .poolSizeCount = asserted_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
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
