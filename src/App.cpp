#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>

#include <imgui.h>

#include "Constants.hpp"
#include "InputHandler.hpp"
#include "Vertex.hpp"
#include "VkUtils.hpp"

using namespace glm;

using std::cerr;
using std::cout;
using std::endl;

namespace {
const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const float CAMERA_FOV = 59.f;
const float CAMERA_NEAR = 0.001f;
const float CAMERA_FAR = 512.f;

vk::DescriptorPool createDescriptorPool(const std::shared_ptr<Device> device,
                                        const uint32_t swapImageCount) {
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount =
            swapImageCount // descriptorCount, camera and mesh instances
    };
    return device->logical().createDescriptorPool(
        vk::DescriptorPoolCreateInfo{.maxSets = poolSize.descriptorCount,
                                     .poolSizeCount = 1,
                                     .pPoolSizes = &poolSize});
}

} // namespace

App::App()
    : _window{WIDTH, HEIGHT, "prosper"}, _device{std::make_shared<Device>(
                                             _window.ptr())},
      _swapConfig{_device, {_window.width(), _window.height()}},
      _imguiRenderer{_device, _window.ptr(), _swapConfig},
      _descriptorPool{createDescriptorPool(_device, _swapConfig.imageCount)},
      _cam{_device, _descriptorPool, _swapConfig.imageCount,
           vk::ShaderStageFlagBits::eVertex |
               vk::ShaderStageFlagBits::eFragment},
      _world{_device, _swapConfig.imageCount,
             resPath("glTF/FlightHelmet/glTF/FlightHelmet.gltf")},
      _swapchain{_device, _swapConfig}, _renderer{_device, _swapConfig,
                                                  _cam.descriptorSetLayout(),
                                                  _world._dsLayouts} {
    _cam.lookAt(vec3{0.25f, 0.2f, 0.75f}, vec3{0.f}, vec3{0.f, 1.f, 0.f});
    _cam.perspective(radians(CAMERA_FOV),
                     _window.width() / static_cast<float>(_window.height()),
                     CAMERA_NEAR, CAMERA_FAR);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        _imageAvailableSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
        _renderFinishedSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
    }

    _swapCommandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = _swapConfig.imageCount});
}

App::~App() {
    for (auto &semaphore : _renderFinishedSemaphores)
        _device->logical().destroy(semaphore);
    for (auto &semaphore : _imageAvailableSemaphores)
        _device->logical().destroy(semaphore);
    _device->logical().destroy(_descriptorPool);
}

void App::run() {
    while (_window.open()) {
        _window.startFrame();

        const auto &mouse = InputHandler::instance().mouse();
        if (mouse.leftDown && mouse.currentPos != mouse.lastPos) {
            _cam.orbit(mouse.currentPos, mouse.lastPos,
                       vec2(_window.width(), _window.height()) / 2.f);
        }
        if (mouse.rightDown && mouse.currentPos != mouse.lastPos) {
            _cam.scaleOrbit(mouse.currentPos.y, mouse.lastPos.y,
                            _window.height() / 2.f);
        }
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    _device->logical().waitIdle();
}

void App::recreateSwapchainAndRelated() {
    while (_window.width() == 0 && _window.height() == 0) {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device->logical().waitIdle();

    _swapConfig = SwapchainConfig{_device, {_window.width(), _window.height()}};

    _swapchain = Swapchain{_device, _swapConfig};
    _renderer.recreateSwapchainRelated(_swapConfig, _cam.descriptorSetLayout(),
                                       _world._dsLayouts);

    _cam.perspective(radians(CAMERA_FOV),
                     _window.width() / static_cast<float>(_window.height()),
                     CAMERA_NEAR, CAMERA_FAR);
}

void App::drawFrame() {
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t nextFrame = _swapchain.nextFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&] {
        const auto imageAvailable = _imageAvailableSemaphores[nextFrame];
        auto nextImage = _swapchain.acquireNextImage(imageAvailable);
        while (!nextImage.has_value()) {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(imageAvailable);
        }

        return nextImage.value();
    }();

    _imguiRenderer.startFrame();

    {
        ImGui::Begin("Stats");

        ImGui::Text("%.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::End();
    }

    const vk::Rect2D renderArea{.offset = {0, 0},
                                .extent = _swapchain.extent()};
    const auto &rendererOutput =
        _renderer.drawFrame(_world, _cam, renderArea, nextImage);

    std::vector<vk::CommandBuffer> commandBuffers = {
        rendererOutput.commandBuffer};

    commandBuffers.push_back(
        _imguiRenderer.endFrame(rendererOutput.image, nextImage));

    {
        // Blit to support different internal rendering resolution (and color
        // format?) the future
        const auto &swapImage = _swapchain.image(nextImage);

        const auto commandBuffer = _swapCommandBuffers[nextImage];
        commandBuffer.reset();

        commandBuffer.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        transitionImageLayout(commandBuffer, rendererOutput.image.handle,
                              vk::ImageSubresourceRange{
                                  .aspectMask = vk::ImageAspectFlagBits::eColor,
                                  .baseMipLevel = 0,
                                  .levelCount = 1,
                                  .baseArrayLayer = 0,
                                  .layerCount = 1},
                              vk::ImageLayout::eColorAttachmentOptimal,
                              vk::ImageLayout::eTransferSrcOptimal,
                              vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::AccessFlagBits::eTransferRead,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits::eTransfer);

        transitionImageLayout(
            commandBuffer, swapImage.handle, swapImage.subresourceRange,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer);

        {
            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};
            const std::array<vk::Offset3D, 2> offsets{
                {{0},
                 {static_cast<int32_t>(_swapConfig.extent.width),
                  static_cast<int32_t>(_swapConfig.extent.height), 1}}};
            const auto fboBlit = vk::ImageBlit{
                .srcSubresource = layers,
                .srcOffsets = offsets,
                .dstSubresource = layers,
                .dstOffsets = offsets,
            };
            commandBuffer.blitImage(rendererOutput.image.handle,
                                    vk::ImageLayout::eTransferSrcOptimal,
                                    swapImage.handle,
                                    vk::ImageLayout::eTransferDstOptimal, 1,
                                    &fboBlit, vk::Filter::eLinear);
        }

        transitionImageLayout(
            commandBuffer, swapImage.handle, swapImage.subresourceRange,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::ePresentSrcKHR, vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eMemoryRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer);

        commandBuffer.end();

        commandBuffers.push_back(commandBuffer);
    }

    // Submit queue
    const std::array<vk::Semaphore, 1> waitSemaphores = {
        _imageAvailableSemaphores[nextFrame]};
    const std::array<vk::PipelineStageFlags, 1> waitStages = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const std::array<vk::Semaphore, 1> signalSemaphores = {
        _renderFinishedSemaphores[nextFrame]};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        .pCommandBuffers = commandBuffers.data(),
        .signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data()};

    checkSuccess(_device->graphicsQueue().submit(1, &submitInfo,
                                                 _swapchain.currentFence()),
                 "submit");

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();
}
