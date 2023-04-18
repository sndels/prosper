#include "ImGuiRenderer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <wheels/containers/static_array.hpp>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace wheels;

namespace
{

constexpr void checkSuccessImGui(VkResult err)
{
    checkSuccess(static_cast<vk::Result>(err), "ImGui");
}

} // namespace

ImGuiRenderer::ImGuiRenderer(
    Device *device, RenderResources *resources, GLFWwindow *window,
    const SwapchainConfig &swapConfig)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(window != nullptr);

    printf("Creating ImGuiRenderer\n");

    createDescriptorPool();
    createRenderPass(_resources->images.toneMapped.format);

    ImGui::CreateContext();
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

    recreate();

    auto buffer = _device->beginGraphicsCommands();

    ImGui_ImplVulkan_CreateFontsTexture(static_cast<VkCommandBuffer>(buffer));

    _device->endGraphicsCommands(buffer);

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

ImGuiRenderer::~ImGuiRenderer()
{
    destroySwapchainRelated();
    _device->logical().destroy(_renderpass);
    _device->logical().destroy(_descriptorPool);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// NOLINTNEXTLINE could be static, but requires an instance TODO: Singleton?
void ImGuiRenderer::startFrame() const
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiRenderer::endFrame(
    vk::CommandBuffer cb, const vk::Rect2D &renderArea, Profiler *profiler)
{
    assert(profiler != nullptr);

    ImGui::Render();
    ImDrawData *drawData = ImGui::GetDrawData();

    {
        const auto _s = profiler->createCpuGpuScope(cb, "ImGui");

        _resources->images.toneMapped.transition(
            cb,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            });

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

void ImGuiRenderer::createRenderPass(const vk::Format &colorFormat)
{
    const vk::AttachmentDescription attachment = {
        .format = colorFormat,
        .samples = vk::SampleCountFlagBits::e1,
        // Assume this works on a populated target
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::AttachmentReference color_attachment = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass = {
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
    _device->logical().destroy(_fbo);
}

void ImGuiRenderer::recreate()
{
    destroySwapchainRelated();

    const auto &image = _resources->images.toneMapped;
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
    const StaticArray poolSizes{
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
    };
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets * asserted_cast<uint32_t>(poolSizes.size()),
            .poolSizeCount = asserted_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });
}
