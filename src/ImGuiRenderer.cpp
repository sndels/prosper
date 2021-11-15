#include "ImGuiRenderer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "VkUtils.hpp"

namespace
{

void checkSuccessImGui(VkResult err)
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
    createDescriptorPool(swapConfig);
    createRenderPass(swapConfig.surfaceFormat.format);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, false);
    ImGui_ImplVulkan_InitInfo init_info = {
        .Instance = _device->instance(),
        .PhysicalDevice = _device->physical(),
        .Device = _device->logical(),
        .QueueFamily = _device->queueFamilies().graphicsFamily.value(),
        .Queue = _device->graphicsQueue(),
        .PipelineCache = vk::PipelineCache{},
        .DescriptorPool = _descriptorPool,
        .MinImageCount = swapConfig.imageCount,
        .ImageCount = swapConfig.imageCount,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        // TODO: Pass in VMA callbacks?
        .CheckVkResultFn = checkSuccessImGui};
    ImGui_ImplVulkan_Init(&init_info, _renderpass);

    recreateSwapchainRelated(swapConfig);

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

void ImGuiRenderer::startFrame() const
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

vk::CommandBuffer ImGuiRenderer::endFrame(
    const vk::Rect2D &renderArea, const uint32_t nextImage)
{
    ImGui::Render();
    ImDrawData *drawData = ImGui::GetDrawData();

    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    buffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = _renderpass,
            .framebuffer = _fbo,
            .renderArea = renderArea},
        vk::SubpassContents::eInline);

    ImGui_ImplVulkan_RenderDrawData(drawData, buffer);

    buffer.endRenderPass();
    buffer.end();

    return buffer;
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
        // Don't do automagical transitions to make this more robust
        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::AttachmentReference color_attachment = {
        .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    };
    vk::SubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
    };
    _renderpass = _device->logical().createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    });

    _device->logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eRenderPass,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkRenderPass>(_renderpass)),
            .pObjectName = "ImGui"});
}

void ImGuiRenderer::destroySwapchainRelated()
{
    if (_commandBuffers.size() > 0)
    {
        _device->logical().freeCommandBuffers(
            _device->graphicsPool(),
            static_cast<uint32_t>(_commandBuffers.size()),
            _commandBuffers.data());
    }
    _device->logical().destroy(_fbo);
}

void ImGuiRenderer::recreateSwapchainRelated(const SwapchainConfig &swapConfig)
{
    destroySwapchainRelated();

    const auto &image = _resources->images.sceneColor;
    _fbo = _device->logical().createFramebuffer(vk::FramebufferCreateInfo{
        .renderPass = _renderpass,
        .attachmentCount = 1,
        .pAttachments = &image.view,
        .width = image.extent.width,
        .height = image.extent.height,
        .layers = 1,
    });

    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount});
}

void ImGuiRenderer::createDescriptorPool(const SwapchainConfig &swapConfig)
{
    const uint32_t maxSets = 1000;
    const std::array<vk::DescriptorPoolSize, 11> poolSizes{{
        {.type = vk::DescriptorType::eSampler, .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eSampledImage, .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eStorageImage, .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eUniformTexelBuffer,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eStorageTexelBuffer,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eStorageBuffer,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eUniformBufferDynamic,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eStorageBufferDynamic,
         .descriptorCount = maxSets},
        {.type = vk::DescriptorType::eInputAttachment,
         .descriptorCount = maxSets},
    }};
    _descriptorPool =
        _device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = maxSets * static_cast<uint32_t>(poolSizes.size()),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()});
}
