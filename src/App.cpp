#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>

#include <glm/gtx/transform.hpp>
#include <imgui.h>

#include "InputHandler.hpp"
#include "Utils.hpp"
#include "Vertex.hpp"
#include "VkUtils.hpp"

using namespace glm;

namespace
{
const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

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
            .commandBufferCount = swapImageCount,
        });
}

vk::DescriptorPool createSwapchainRelatedDescriptorPool(
    const Device *device, const uint32_t swapImageCount)
{
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eStorageImage,
        .descriptorCount =
            (2 + 2) *
            swapImageCount, // tonemap input/output, light clustering outputs
    };

    return device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets = poolSize.descriptorCount,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    });
}

RenderResources::DescriptorPools createDescriptorPools(
    const Device *device, const uint32_t swapImageCount)
{
    RenderResources::DescriptorPools pools;
    {
        const vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = swapImageCount, // camera uniforms
        };
        pools.constant =
            device->logical().createDescriptorPool(vk::DescriptorPoolCreateInfo{
                .maxSets = poolSize.descriptorCount,
                .poolSizeCount = 1,
                .pPoolSizes = &poolSize,
            });
    }

    pools.swapchainRelated =
        createSwapchainRelatedDescriptorPool(device, swapImageCount);

    return pools;
}
} // namespace

App::App(const std::filesystem::path & scene, bool enableDebugLayers)
: _window{{WIDTH, HEIGHT}, "prosper"}
, _device{_window.ptr(), enableDebugLayers}
, _swapConfig{&_device, {_window.width(), _window.height()}}
, _swapchain{&_device, _swapConfig}
, _swapCommandBuffers{allocateSwapCommandBuffers(&_device, _swapConfig.imageCount)}
, _resources{
    .descriptorPools =
        createDescriptorPools(&_device, _swapConfig.imageCount)}
, _cam{&_device, _resources.descriptorPools.constant, _swapConfig.imageCount,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
    | vk::ShaderStageFlagBits::eCompute|
    vk::ShaderStageFlagBits::eRaygenKHR}
, _world{
    &_device, _swapConfig.imageCount,
    scene}
, _lightClustering{

      &_device, &_resources, _swapConfig, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _renderer{
      &_device, &_resources, _swapConfig, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _rtRenderer{
      &_device, &_resources, _swapConfig, _cam.descriptorSetLayout(),_world._dsLayouts}
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

        recompileShaders();

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
    _lightClustering.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _renderer.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _rtRenderer.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _transparentsRenderer.recreateSwapchainRelated(
        _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
    _skyboxRenderer.recreateSwapchainRelated(_swapConfig, _world._dsLayouts);
    _toneMap.recreateSwapchainRelated(_swapConfig);
    _imguiRenderer.recreateSwapchainRelated(_swapConfig);

    _cam.perspective(
        PerspectiveParameters{
            .fov = radians(CAMERA_FOV),
            .zN = CAMERA_NEAR,
            .zF = CAMERA_FAR,
        },
        _window.width() / static_cast<float>(_window.height()));
}

void App::recompileShaders()
{
    if (_recompileShaders)
    {
        // Wait for resources to be out of use
        _device.logical().waitIdle();

        fprintf(stderr, "Recompiling shaders\n");

        Timer t;

        _lightClustering.recompileShaders(
            _cam.descriptorSetLayout(), _world._dsLayouts);
        _renderer.recompileShaders(
            _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
        _rtRenderer.recompileShaders(
            _cam.descriptorSetLayout(), _world._dsLayouts);
        _transparentsRenderer.recompileShaders(
            _swapConfig, _cam.descriptorSetLayout(), _world._dsLayouts);
        _skyboxRenderer.recompileShaders(_swapConfig, _world._dsLayouts);
        _toneMap.recompileShaders();

        fprintf(stderr, "Shaders recompiled in %.2fs\n", t.getSeconds());

        _recompileShaders = false;
    }
}

void App::handleMouseGestures()
{
    // Gestures adapted from Max Liani
    // https://maxliani.wordpress.com/2021/06/08/offline-to-realtime-camera-manipulation/

    const auto &gesture = InputHandler::instance().mouseGesture();
    if (gesture)
    {
        if (gesture->type == MouseGestureType::TrackBall)
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
        else if (gesture->type == MouseGestureType::TrackPlane)
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
        else if (gesture->type == MouseGestureType::TrackZoom)
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
        ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_Appearing);
        ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Text(
            "%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
            ImGui::GetIO().Framerate);

        ImGui::Checkbox("Limit FPS", &_useFpsLimit);
        if (_useFpsLimit)
        {
            ImGui::DragInt("##FPS limit value", &_fpsLimit, 5.f, 30, 250);
        }

        _recompileShaders = ImGui::Button("Recompile shaders");

        ImGui::Checkbox("Render RT", &_renderRT);

        ImGui::End();
    }

    const vk::Rect2D renderArea{
        .offset = {0, 0},
        .extent = _swapchain.extent(),
    };

    std::vector<vk::CommandBuffer> commandBuffers;

    assert(
        renderArea.offset.x == 0 && renderArea.offset.y == 0 &&
        "Camera update assumes no render offset");
    _cam.updateBuffer(
        nextImage, uvec2{renderArea.extent.width, renderArea.extent.height});
    _world.updateUniformBuffers(_cam, nextImage);

    const auto &scene = _world.currentScene();

    commandBuffers.push_back(_lightClustering.recordCommandBuffer(
        scene, _cam, renderArea, nextImage));

    if (_renderRT)
    {
        commandBuffers.push_back(_rtRenderer.recordCommandBuffer(
            _world, _cam, renderArea, nextImage));
    }
    else
    {
        commandBuffers.push_back(
            _renderer.recordCommandBuffer(_world, _cam, renderArea, nextImage));

        commandBuffers.push_back(_transparentsRenderer.recordCommandBuffer(
            _world, _cam, renderArea, nextImage));

        commandBuffers.push_back(
            _skyboxRenderer.recordCommandBuffer(_world, renderArea, nextImage));
    }

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

        const std::array<vk::ImageMemoryBarrier2, 2> barriers{
            _resources.images.toneMapped.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                .accessMask = vk::AccessFlagBits2::eTransferRead,
                .layout = vk::ImageLayout::eTransferSrcOptimal,
            }),
            vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                .srcAccessMask = vk::AccessFlags2{},
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapImage.handle,
                .subresourceRange = swapImage.subresourceRange,
            },
        };

        commandBuffer.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        {
            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};
            const std::array<vk::Offset3D, 2> offsets{{
                vk::Offset3D{0},
                vk::Offset3D{
                    asserted_cast<int32_t>(_swapConfig.extent.width),
                    asserted_cast<int32_t>(_swapConfig.extent.height),
                    1,
                },
            }};
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
            const vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::ePresentSrcKHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapImage.handle,
                .subresourceRange = swapImage.subresourceRange,
            };

            commandBuffer.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            });
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
        .waitSemaphoreCount = asserted_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = asserted_cast<uint32_t>(commandBuffers.size()),
        .pCommandBuffers = commandBuffers.data(),
        .signalSemaphoreCount =
            asserted_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    checkSuccess(
        _device.graphicsQueue().submit(
            1, &submitInfo, _swapchain.currentFence()),
        "submit");

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();
}
