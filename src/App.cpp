#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/gtx/transform.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <imgui.h>

#include "InputHandler.hpp"
#include "Utils.hpp"
#include "Vertex.hpp"
#include "VkUtils.hpp"

using namespace glm;

using std::cerr;
using std::cout;
using std::endl;

namespace
{
const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const float CAMERA_FOV = 59.f;
const float CAMERA_NEAR = 0.001f;
const float CAMERA_FAR = 512.f;

std::vector<vk::CommandBuffer> allocateSwapCommandBuffers(
    Device *device, const uint32_t swapImageCount)
{
    return device->logical().allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{
            .commandPool = device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapImageCount});
}

vk::DescriptorPool createSwapchainRelatedDescriptorPool(
    const Device *device, const uint32_t swapImageCount)
{
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eStorageImage,
        .descriptorCount = 2 * swapImageCount // tonemap input/output
    };

    return device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets = poolSize.descriptorCount,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize});
}

RenderResources::DescriptorPools createDescriptorPools(
    const Device *device, const uint32_t swapImageCount)
{
    RenderResources::DescriptorPools pools;
    {
        const vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = swapImageCount // camera uniforms
        };
        pools.constant =
            device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
                .maxSets = poolSize.descriptorCount,
                .poolSizeCount = 1,
                .pPoolSizes = &poolSize});
    }

    pools.swapchainRelated =
        createSwapchainRelatedDescriptorPool(device, swapImageCount);

    return pools;
}
} // namespace

App::App(const std::filesystem::path & scene)
: _window{WIDTH, HEIGHT, "prosper"}
, _device{_window.ptr()}
, _swapConfig{&_device, {_window.width(), _window.height()}}
, _swapchain{&_device, _swapConfig}
, _swapCommandBuffers{allocateSwapCommandBuffers(&_device, _swapConfig.imageCount)}
, _resources{
    .descriptorPools =
        createDescriptorPools(&_device, _swapConfig.imageCount)}
, _cam{&_device, _resources.descriptorPools.constant, _swapConfig.imageCount,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment}
, _world{
    &_device, _swapConfig.imageCount,
    scene}
, _renderer{
      &_device, &_resources, _swapConfig, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _transparentsRenderer{
      &_device, &_resources, _swapConfig, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _skyboxRenderer{
      &_device, &_resources, _swapConfig,
      _world._dsLayouts}
, _toneMap{&_device, &_resources, _swapConfig}
, _imguiRenderer{&_device, &_resources, _window.ptr(), _swapConfig}
{
    _cam.init(_world._scenes[_world._currentScene].camera);
    _cam.perspective(_window.width() / static_cast<float>(_window.height()));

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _imageAvailableSemaphores.push_back(
            _device.logical().createSemaphore(vk::SemaphoreCreateInfo{}));
        _renderFinishedSemaphores.push_back(
            _device.logical().createSemaphore(vk::SemaphoreCreateInfo{}));
    }
}

App::~App()
{
    for (auto &semaphore : _renderFinishedSemaphores)
        _device.logical().destroy(semaphore);
    for (auto &semaphore : _imageAvailableSemaphores)
        _device.logical().destroy(semaphore);
    _device.logical().destroy(_resources.descriptorPools.constant);
    _device.logical().destroy(_resources.descriptorPools.swapchainRelated);
}

void App::run()
{
    while (_window.open())
    {
        _window.startFrame();

        handleMouseGestures();

        drawFrame();

        InputHandler::instance().clearSingleFrameGestures();
    }

    // Wait for in flight rendering actions to finish
    _device.logical().waitIdle();
}

void App::recreateSwapchainAndRelated()
{
    while (_window.width() == 0 && _window.height() == 0)
    {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device.logical().waitIdle();

    _swapConfig =
        SwapchainConfig{&_device, {_window.width(), _window.height()}};

    _swapchain.recreate(_swapConfig);

    // We could free and recreate the individual sets but destroying the pool is
    // cleaner
    _device.logical().destroyDescriptorPool(
        _resources.descriptorPools.swapchainRelated);
    _resources.descriptorPools.swapchainRelated =
        createSwapchainRelatedDescriptorPool(&_device, _swapConfig.imageCount);

    // NOTE: These need to be in the order that RenderResources contents are
    // written to!
    _renderer.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _transparentsRenderer.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _skyboxRenderer.recreateSwapchainRelated(_swapConfig, _world._dsLayouts);
    _toneMap.recreateSwapchainRelated(_swapConfig);
    _imguiRenderer.recreateSwapchainRelated(_swapConfig);

    _cam.perspective(
        radians(CAMERA_FOV),
        _window.width() / static_cast<float>(_window.height()), CAMERA_NEAR,
        CAMERA_FAR);
}

void App::handleMouseGestures()
{
    // Gestures adapted from Max Liani
    // https://maxliani.wordpress.com/2021/06/08/offline-to-realtime-camera-manipulation/

    const auto &gesture = InputHandler::instance().mouseGesture();
    if (gesture)
    {
        using enum MouseGestureType;
        if (gesture->type == TrackBall)
        {

            const auto dragScale = 1.f / 400.f;
            const auto drag =
                (gesture->currentPos - gesture->startPos) * dragScale;

            const auto params = _cam.parameters();
            const auto fromTarget = params.eye - params.target;

            const auto horizontalRotatedFromTarget =
                mat3(rotate(-drag.x, params.up)) * fromTarget;

            const auto right =
                normalize(cross(horizontalRotatedFromTarget, params.up));

            const auto newFromTarget =
                mat3(rotate(drag.y, right)) * horizontalRotatedFromTarget;
            const auto flipUp =
                dot(right, cross(newFromTarget, params.up)) < 0.0;

            _cam.offset = CameraOffset{
                .eye = newFromTarget - fromTarget,
                .flipUp = flipUp,
            };
        }
        else if (gesture->type == TrackPlane)
        {
            const auto params = _cam.parameters();
            const auto from_target = params.eye - params.target;
            const auto dist_target = length(from_target);

            // TODO: Adjust for aspect ratio difference between film and window
            const auto drag_scale = [&]
            {
                auto tanHalfFov = tan(params.fov * 0.5f);
                return dist_target * tanHalfFov /
                       (static_cast<float>(_window.height()) * 0.5f);
            }();
            const auto drag =
                (gesture->currentPos - gesture->startPos) * drag_scale;

            const auto right = normalize(cross(from_target, params.up));
            const auto cam_up = normalize(cross(right, from_target));

            const auto offset = right * (drag.x) + cam_up * (drag.y);

            _cam.offset = CameraOffset{
                .eye = offset,
                .target = offset,
            };
        }
        else if (gesture->type == TrackZoom)
        {
            if (!_cam.offset)
            {
                const auto &params = _cam.parameters();

                const auto to_target = params.target - params.eye;
                const auto dist_target = length(to_target);
                const auto fwd = to_target / dist_target;

                const auto scroll_scale = dist_target * 0.1f;

                const auto offset = CameraOffset{
                    .eye = fwd * gesture->verticalScroll * scroll_scale,
                };

                // Make sure we don't get too close to get stuck
                const auto offsetParams = params.apply(offset);
                if (all(greaterThan(
                        abs(offsetParams.eye - params.target),
                        vec3(compMax(
                            0.01f * max(offsetParams.eye, params.target))))))
                {
                    _cam.offset = offset;
                }
            }
        }
        else
            throw std::runtime_error("Unknown mouse gesture");
    }
    else
    {
        if (_cam.offset)
        {
            _cam.applyOffset();
        }
    }
}

void App::drawFrame()
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t nextFrame = _swapchain.nextFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&]
    {
        const auto imageAvailable = _imageAvailableSemaphores[nextFrame];
        auto nextImage = _swapchain.acquireNextImage(imageAvailable);
        while (!nextImage.has_value())
        {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(imageAvailable);
        }

        return nextImage.value();
    }();

    // Enforce fps cap by spinlocking to have any hope to be somewhat consistent
    // Note that this is always based on the previous frame so it only limits
    // fps and doesn't help actual frame timing
    {
        const float minDt =
            _useFpsLimit ? 1.f / static_cast<float>(_fpsLimit) : 0.f;
        while (_frameTimer.getSeconds() < minDt)
        {
            ;
        }
    }
    _frameTimer.reset();

    _imguiRenderer.startFrame();

    {
        ImGui::Begin("Stats");

        ImGui::Text(
            "%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
            ImGui::GetIO().Framerate);

        ImGui::Checkbox("Limit FPS", &_useFpsLimit);
        if (_useFpsLimit)
        {
            ImGui::DragInt("##FPS limit value", &_fpsLimit, 5.f, 30, 250);
        }

        ImGui::End();
    }

    const vk::Rect2D renderArea{
        .offset = {0, 0}, .extent = _swapchain.extent()};

    std::vector<vk::CommandBuffer> commandBuffers;

    _cam.updateBuffer(nextImage);

    commandBuffers.push_back(
        _renderer.execute(_world, _cam, renderArea, nextImage));

    commandBuffers.push_back(
        _transparentsRenderer.execute(_world, _cam, renderArea, nextImage));

    commandBuffers.push_back(
        _skyboxRenderer.execute(_world, _cam, renderArea, nextImage));

    commandBuffers.push_back(_toneMap.execute(nextImage));

    commandBuffers.push_back(_imguiRenderer.endFrame(renderArea, nextImage));

    {
        // Blit to support different internal rendering resolution (and color
        // format?) the future
        const auto &swapImage = _swapchain.image(nextImage);

        const auto commandBuffer = _swapCommandBuffers[nextImage];
        commandBuffer.reset();

        commandBuffer.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        _resources.images.toneMapped.transitionBarrier(
            commandBuffer, vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eTransfer);
        {
            vk::ImageMemoryBarrier barrier{
                .srcAccessMask = vk::AccessFlags{},
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapImage.handle,
                .subresourceRange = swapImage.subresourceRange};

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, 0,
                nullptr, 0, nullptr, 1, &barrier);
        }

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
            commandBuffer.blitImage(
                _resources.images.toneMapped.handle,
                vk::ImageLayout::eTransferSrcOptimal, swapImage.handle,
                vk::ImageLayout::eTransferDstOptimal, 1, &fboBlit,
                vk::Filter::eLinear);
        }

        {
            const vk::ImageMemoryBarrier barrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::ePresentSrcKHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapImage.handle,
                .subresourceRange = swapImage.subresourceRange};

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, 0,
                nullptr, 0, nullptr, 1, &barrier);
        }

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

    checkSuccess(
        _device.graphicsQueue().submit(
            1, &submitInfo, _swapchain.currentFence()),
        "submit");

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();
}
