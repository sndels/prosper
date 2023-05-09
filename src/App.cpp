#include "App.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <glm/gtx/transform.hpp>
#include <imgui.h>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/string.hpp>

#include "InputHandler.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;

constexpr float CAMERA_FOV = 59.f;
constexpr float CAMERA_NEAR = 0.001f;
constexpr float CAMERA_FAR = 512.f;

StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> allocateCommandBuffers(
    Device *device)
{
    StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> ret;
    ret.resize(MAX_FRAMES_IN_FLIGHT);

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = device->graphicsPool(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    checkSuccess(
        device->logical().allocateCommandBuffers(&allocInfo, ret.data()),
        "Failed to allocate command buffers");

    return ret;
}

} // namespace

// TODO: Fix this file so that clang-format works

App::App(ScopedScratch scopeAlloc, const std::filesystem::path & scene, bool enableDebugLayers)
: _window{Pair<uint32_t, uint32_t>{WIDTH, HEIGHT}, "prosper"}
, _device{scopeAlloc.child_scope(), _window.ptr(), enableDebugLayers}
, _swapchain{&_device, SwapchainConfig{scopeAlloc.child_scope(),&_device, {_window.width(), _window.height()}}}
, _commandBuffers{allocateCommandBuffers(&_device)}
, _viewportExtent{_swapchain.config().extent}
, _resources{_generalAlloc,&_device}
, _cam{&_device, &_resources}
, _world{ scopeAlloc.child_scope(), &_device, scene}
, _lightClustering{
    scopeAlloc.child_scope(),
      &_device, &_resources, _viewportExtent, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _renderer{
    scopeAlloc.child_scope(),
      &_device, &_resources, _viewportExtent, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _gbufferRenderer{
    scopeAlloc.child_scope(),
      &_device, &_resources, _viewportExtent, _cam.descriptorSetLayout(),
      _world._dsLayouts}
, _deferredShading{
    scopeAlloc.child_scope(),
      &_device, &_resources, _cam.descriptorSetLayout(), _world._dsLayouts}
, _rtRenderer{
    scopeAlloc.child_scope(), &_device, &_resources, _cam.descriptorSetLayout(), _world._dsLayouts}
, _skyboxRenderer{
    scopeAlloc.child_scope(),
      &_device, &_resources, _viewportExtent,
      _world._dsLayouts}
, _debugRenderer{
    scopeAlloc.child_scope(),
    &_device, &_resources, _viewportExtent, _cam.descriptorSetLayout()}
, _toneMap{
    scopeAlloc.child_scope(),
    &_device, &_resources, _viewportExtent}
, _imguiRenderer{&_device, &_resources,_swapchain.config().extent, _window.ptr(), _swapchain.config()}
,_profiler{_generalAlloc, &_device}
,_recompileTime{std::chrono::file_clock::now()}
{
    printf("GPU pass init took %.2fs\n", _gpuPassesInitTimer.getSeconds());

    const auto &allocs = _device.memoryAllocations();
    printf("Active GPU allocations:\n");
    printf(
        "  Buffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.buffers / 1024 / 1024));
    printf(
        "  TexelBuffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.texelBuffers / 1024 / 1024));
    printf(
        "  Images: %uMB\n",
        asserted_cast<uint32_t>(allocs.images / 1024 / 1024));

    _cam.init(_world._scenes[_world._currentScene].camera);
    _cam.perspective(
        _viewportExtent.width / static_cast<float>(_viewportExtent.height));

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
}

void App::run()
{
    LinearAllocator scopeBackingAlloc{megabytes(64)};
    while (_window.open())
    {
        _profiler.startCpuFrame();

        scopeBackingAlloc.reset();
        ScopedScratch scopeAlloc{scopeBackingAlloc};

        {
            const auto _s = _profiler.createCpuScope("Window::startFrame");
            _window.startFrame();
        }

        handleMouseGestures();

        recompileShaders(scopeAlloc.child_scope());

        drawFrame(scopeAlloc.child_scope());

        InputHandler::instance().clearSingleFrameGestures();
        _cam.clearChangedThisFrame();

        _profiler.endCpuFrame();
    }

    // Wait for in flight rendering actions to finish
    _device.logical().waitIdle();
}

void App::recreateSwapchainAndRelated(wheels::ScopedScratch scopeAlloc)
{
    while (_window.width() == 0 && _window.height() == 0)
    {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device.logical().waitIdle();

    { // Drop the config as we should always use swapchain's active config
        SwapchainConfig config{
            scopeAlloc.child_scope(),
            &_device,
            {_window.width(), _window.height()}};
        _swapchain.recreate(config);
    }

    const ImVec2 viewportSize = _imguiRenderer.centerAreaSize();
    _viewportExtent = vk::Extent2D{
        asserted_cast<uint32_t>(viewportSize.x),
        asserted_cast<uint32_t>(viewportSize.y),
    };

    // We could free and recreate the individual sets but reseting the pools is
    // cleaner
    _resources.descriptorAllocator.resetPools();

    _cam.recreate();

    // NOTE: These need to be in the order that RenderResources contents are
    // written to!
    _lightClustering.recreate(
        _viewportExtent, _cam.descriptorSetLayout(), _world._dsLayouts);
    _renderer.recreate(
        _viewportExtent, _cam.descriptorSetLayout(), _world._dsLayouts);
    _gbufferRenderer.recreate(
        _viewportExtent, _cam.descriptorSetLayout(), _world._dsLayouts);
    _deferredShading.recreate(_cam.descriptorSetLayout(), _world._dsLayouts);
    _rtRenderer.recreate(
        scopeAlloc.child_scope(), _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _skyboxRenderer.recreate(_viewportExtent, _world._dsLayouts);
    _debugRenderer.recreate(_viewportExtent, _cam.descriptorSetLayout());
    _toneMap.recreate(_viewportExtent);
    _imguiRenderer.recreate(_swapchain.config().extent);

    _cam.perspective(
        PerspectiveParameters{
            .fov = radians(CAMERA_FOV),
            .zN = CAMERA_NEAR,
            .zF = CAMERA_FAR,
        },
        _viewportExtent.width / static_cast<float>(_viewportExtent.height));
}

void App::recompileShaders(ScopedScratch scopeAlloc)
{
    const auto _s = _profiler.createCpuScope("App::recompileShaders");

    if (!_recompileShaders)
        return;

    Timer checkTime;
    bool shadersChanged = false;
    auto shadersIterator =
        std::filesystem::recursive_directory_iterator(resPath("shader"));
    for (const auto &entry : shadersIterator)
    {
        if (entry.last_write_time() > _recompileTime)
        {
            shadersChanged = true;
            break;
        }
    }
    if (checkTime.getSeconds() > 0.001f)
        fprintf(
            stderr, "Shader timestamp check is laggy: %.1fms\n",
            checkTime.getSeconds() * 1000.f);

    if (!shadersChanged)
        return;

    // Wait for resources to be out of use
    _device.logical().waitIdle();

    printf("Recompiling shaders\n");

    Timer t;

    _lightClustering.recompileShaders(
        scopeAlloc.child_scope(), _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _renderer.recompileShaders(
        scopeAlloc.child_scope(), _viewportExtent, _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _gbufferRenderer.recompileShaders(
        scopeAlloc.child_scope(), _viewportExtent, _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _deferredShading.recompileShaders(
        scopeAlloc.child_scope(), _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _rtRenderer.recompileShaders(
        scopeAlloc.child_scope(), _cam.descriptorSetLayout(),
        _world._dsLayouts);
    _skyboxRenderer.recompileShaders(
        scopeAlloc.child_scope(), _viewportExtent, _world._dsLayouts);
    _debugRenderer.recompileShaders(
        scopeAlloc.child_scope(), _viewportExtent, _cam.descriptorSetLayout());
    _toneMap.recompileShaders(scopeAlloc.child_scope());

    printf("Shaders recompiled in %.2fs\n", t.getSeconds());

    _recompileTime = std::chrono::file_clock::now();
}

void App::handleMouseGestures()
{
    const auto _s = _profiler.createCpuScope("App::handleMouseGestures");

    // Gestures adapted from Max Liani
    // https://maxliani.wordpress.com/2021/06/08/offline-to-realtime-camera-manipulation/

    const auto &gesture = InputHandler::instance().mouseGesture();
    if (gesture.has_value())
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
                       (static_cast<float>(_viewportExtent.height) * 0.5f);
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
            if (!_cam.offset.has_value())
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
        if (_cam.offset.has_value())
        {
            _cam.applyOffset();
        }
    }
}

void App::drawFrame(ScopedScratch scopeAlloc)
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
            // Wait on the acquire semaphore to have it properly unsignaled.
            // Validation would otherwise complain on next acquire below even
            // with the wait for idle.
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &imageAvailable,
            };

            checkSuccess(
                _device.graphicsQueue().submit(1, &submitInfo, vk::Fence{}),
                "recreate_swap_dummy_submit");

            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated(scopeAlloc.child_scope());
            nextImage = _swapchain.acquireNextImage(imageAvailable);
        }

        return *nextImage;
    }();

    _profiler.startGpuFrame(nextFrame);

    const auto profilerDatas = _profiler.getPreviousData(scopeAlloc);

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

    bool rtPickedThisFrame = false;
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

        ImGui::Checkbox("Recompile shaders", &_recompileShaders);

        rtPickedThisFrame =
            ImGui::Checkbox("Render RT", &_renderRT) && _renderRT;

        if (!_renderRT)
            ImGui::Checkbox("Use deferred shading", &_renderDeferred);

        ImGui::End();
    }

    {
        ImGui::SetNextWindowPos(
            ImVec2{1920.f - 300.f, 60.f}, ImGuiCond_Appearing);
        ImGui::Begin("Profiling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        {
            // Having names longer than 255 characters is an error
            uint8_t longestNameLength = 0;
            for (const auto &t : profilerDatas)
                if (t.name.size() > longestNameLength)
                    longestNameLength = asserted_cast<uint8_t>(t.name.size());

            // Double the maximum name length for headroom
            String tmp{scopeAlloc};
            tmp.resize(longestNameLength * 2);
            const auto leftJustified =
                [&tmp, longestNameLength](StrSpan str, uint8_t extraWidth = 0)
            {
                // No realloc, please
                assert(longestNameLength + extraWidth <= tmp.size());

                tmp.clear();
                tmp.extend(str);
                tmp.resize(
                    std::max(
                        str.size(),
                        static_cast<size_t>(longestNameLength + extraWidth)),
                    ' ');

                return tmp.c_str();
            };

            // Force minimum window size with whitespace
            if (ImGui::CollapsingHeader(
                    leftJustified("GPU"), ImGuiTreeNodeFlags_DefaultOpen))
            {
                for (const auto &t : profilerDatas)
                    if (t.gpuMillis >= 0.f)
                        ImGui::Text(
                            "%s %.3fms", leftJustified(t.name), t.gpuMillis);
                if (!profilerDatas.empty())
                {
                    static int scopeIndex = 0;
                    const char *comboTitle =
                        profilerDatas[scopeIndex].gpuMillis < 0.f
                            ? "##EmptyGPUScopeTitle"
                            : profilerDatas[scopeIndex].name.data();
                    if (ImGui::BeginCombo("##GPUScopeData", comboTitle, 0))
                    {
                        for (int n = 0;
                             n < asserted_cast<int>(profilerDatas.size()); n++)
                        {
                            // Only have scopes that have gpu data
                            if (profilerDatas[n].gpuMillis >= 0.f)
                            {
                                const bool selected = scopeIndex == n;
                                if (ImGui::Selectable(
                                        profilerDatas[n].name.data(), selected))
                                    scopeIndex = n;

                                if (selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    if (profilerDatas[scopeIndex].gpuMillis >= 0.f)
                    {
                        const auto &stats = profilerDatas[scopeIndex].stats;
                        const auto &swapExtent = _viewportExtent;
                        const uint32_t pixelCount =
                            swapExtent.width * swapExtent.height;

                        // Stats from AMD's 'D3D12 Right On Queue' GDC2016
                        const float rasterPrimPerPrim =
                            stats.iaPrimitives == 0
                                ? -1.f
                                : static_cast<float>(stats.clipPrimitives) /
                                      static_cast<float>(stats.iaPrimitives);

                        const float fragsPerPrim =
                            stats.clipPrimitives == 0
                                ? -1.f
                                : static_cast<float>(stats.fragInvocations) /
                                      static_cast<float>(stats.clipPrimitives);

                        const float overdraw =
                            stats.fragInvocations == 0
                                ? -1.f
                                : static_cast<float>(stats.fragInvocations) /
                                      static_cast<float>(pixelCount);

                        ImGui::Indent();

                        ImGui::Text(
                            "Raster prim per prim: %.2f", rasterPrimPerPrim);
                        ImGui::Text("Frags per prim: %.2f", fragsPerPrim);
                        ImGui::Text("Overdraw: %.2f", overdraw);

                        ImGui::Unindent();
                    }
                }
            }

            if (ImGui::CollapsingHeader(
                    leftJustified("CPU"), ImGuiTreeNodeFlags_DefaultOpen))
            {
                for (const auto &t : profilerDatas)
                    if (t.cpuMillis >= 0.f)
                        ImGui::Text(
                            "%s %.3fms", leftJustified(t.name), t.cpuMillis);
            }
        }

        ImGui::End();
    }

    const vk::Rect2D renderArea{
        .offset = {0, 0},
        .extent = _viewportExtent,
    };

    assert(
        renderArea.offset.x == 0 && renderArea.offset.y == 0 &&
        "Camera update assumes no render offset");
    _cam.updateBuffer(
        nextFrame, uvec2{renderArea.extent.width, renderArea.extent.height});
    _world.updateUniformBuffers(_cam, nextFrame, scopeAlloc.child_scope());

    const auto &scene = _world.currentScene();

    auto &debugLines = _resources.buffers.debugLines[nextFrame];
    debugLines.reset();
    { // Add debug geom for lights
        constexpr auto debugLineLength = 0.2f;
        constexpr auto debugRed = vec3{1.f, 0.05f, 0.05f};
        constexpr auto debugGreen = vec3{0.05f, 1.f, 0.05f};
        constexpr auto debugBlue = vec3{0.05f, 0.05f, 1.f};
        for (const auto &pl : scene.lights.pointLights.data)
        {
            const auto pos = vec3{pl.position};
            debugLines.addLine(
                pos, pos + vec3{debugLineLength, 0.f, 0.f}, debugRed);
            debugLines.addLine(
                pos, pos + vec3{0.f, debugLineLength, 0.f}, debugGreen);
            debugLines.addLine(
                pos, pos + vec3{0.f, 0.f, debugLineLength}, debugBlue);
        }

        for (const auto &sl : scene.lights.spotLights.data)
        {
            const auto pos = vec3{sl.positionAndAngleOffset};
            const auto fwd = vec3{sl.direction};
            const auto right = abs(1 - sl.direction.y) < 0.1
                                   ? normalize(cross(fwd, vec3{0.f, 0.f, -1.f}))
                                   : normalize(cross(fwd, vec3{0.f, 1.f, 0.f}));
            const auto up = normalize(cross(right, fwd));
            debugLines.addLine(pos, pos + right * debugLineLength, debugRed);
            debugLines.addLine(pos, pos + up * debugLineLength, debugGreen);
            debugLines.addLine(pos, pos + fwd * debugLineLength, debugBlue);
        }
    }

    const auto cb = _commandBuffers[nextFrame];
    cb.reset();

    cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    _lightClustering.record(cb, scene, _cam, renderArea, nextFrame, &_profiler);

    if (_renderRT)
    {
        _rtRenderer.drawUi();
        _rtRenderer.record(
            cb, _world, _cam, renderArea, nextFrame, rtPickedThisFrame,
            &_profiler);
    }
    else
    {

        // Opaque
        if (_renderDeferred)
        {
            _deferredShading.drawUi();
            _gbufferRenderer.record(
                cb, _world, _cam, renderArea, nextFrame, &_profiler);
            _deferredShading.record(cb, _world, _cam, nextFrame, &_profiler);
        }
        else
        {
            _renderer.drawUi();
            _renderer.record(
                cb, _world, _cam, renderArea, nextFrame, false, &_profiler);
        }

        // Transparent
        _renderer.record(
            cb, _world, _cam, renderArea, nextFrame, true, &_profiler);

        _skyboxRenderer.record(cb, _world, renderArea, nextFrame, &_profiler);
    }

    _debugRenderer.record(cb, _cam, renderArea, nextFrame, &_profiler);

    _toneMap.drawUi();
    _toneMap.record(cb, nextFrame, &_profiler);

    { // TODO: Split into function
        // Blit tonemapped into cleared final composite before drawing ui on top
        {
            const StaticArray barriers{{
                _resources.images.toneMapped.transitionBarrier(ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .accessMask = vk::AccessFlagBits2::eTransferRead,
                    .layout = vk::ImageLayout::eTransferSrcOptimal,
                }),
                _resources.images.finalComposite.transitionBarrier(ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .accessMask = vk::AccessFlagBits2::eTransferWrite,
                    .layout = vk::ImageLayout::eTransferDstOptimal,
                }),
            }};

            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount =
                    asserted_cast<uint32_t>(barriers.size()),
                .pImageMemoryBarriers = barriers.data(),
            });
        }

        const vk::ClearColorValue clearColor{0.f, 0.f, 0.f, 0.f};
        const vk::ImageSubresourceRange subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        cb.clearColorImage(
            _resources.images.finalComposite.handle,
            vk::ImageLayout::eTransferDstOptimal, &clearColor, 1,
            &subresourceRange);

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, {}, {},
            {});

        const vk::ImageSubresourceLayers layers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1};

        const std::array srcOffsets{
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{
                asserted_cast<int32_t>(_viewportExtent.width),
                asserted_cast<int32_t>(_viewportExtent.height),
                1,
            },
        };

        const ImVec2 dstOffset = _imguiRenderer.centerAreaOffset();
        const ImVec2 dstSize = _imguiRenderer.centerAreaSize();
        const vk::Extent2D backbufferExtent = _swapchain.config().extent;
        const std::array dstOffsets{
            vk::Offset3D{
                std::min(
                    asserted_cast<int32_t>(dstOffset.x),
                    asserted_cast<int32_t>(backbufferExtent.width - 1)),
                std::min(
                    asserted_cast<int32_t>(dstOffset.y),
                    asserted_cast<int32_t>(backbufferExtent.height - 1)),
                0,
            },
            vk::Offset3D{
                std::min(
                    asserted_cast<int32_t>(dstOffset.x + dstSize.x),
                    asserted_cast<int32_t>(backbufferExtent.width)),
                std::min(
                    asserted_cast<int32_t>(dstOffset.y + dstSize.y),
                    asserted_cast<int32_t>(backbufferExtent.height)),
                1,
            },
        };
        vk::ImageBlit blit = {
            .srcSubresource = layers,
            .srcOffsets = srcOffsets,
            .dstSubresource = layers,
            .dstOffsets = dstOffsets,
        };
        cb.blitImage(
            _resources.images.toneMapped.handle,
            vk::ImageLayout::eTransferSrcOptimal,
            _resources.images.finalComposite.handle,
            vk::ImageLayout::eTransferDstOptimal, 1, &blit,
            vk::Filter::eLinear);
    }

    const vk::Rect2D backbufferArea{
        .offset = {0, 0},
        .extent = _swapchain.config().extent,
    };
    _imguiRenderer.endFrame(cb, backbufferArea, &_profiler);

    {
        // Blit to support different internal rendering resolution (and color
        // format?) the future
        const auto &swapImage = _swapchain.image(nextImage);

        const StaticArray barriers{{
            _resources.images.finalComposite.transitionBarrier(ImageState{
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
        }};

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        {
            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1};

            assert(
                _resources.images.finalComposite.extent.width ==
                swapImage.extent.width);
            assert(
                _resources.images.finalComposite.extent.height ==
                swapImage.extent.height);
            const std::array offsets{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{
                    asserted_cast<int32_t>(_swapchain.config().extent.width),
                    asserted_cast<int32_t>(_swapchain.config().extent.height),
                    1,
                },
            };
            const auto blit = vk::ImageBlit{
                .srcSubresource = layers,
                .srcOffsets = offsets,
                .dstSubresource = layers,
                .dstOffsets = offsets,
            };
            cb.blitImage(
                _resources.images.finalComposite.handle,
                vk::ImageLayout::eTransferSrcOptimal, swapImage.handle,
                vk::ImageLayout::eTransferDstOptimal, 1, &blit,
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

            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            });
        }

        _profiler.endGpuFrame(cb);

        cb.end();
    }

    // Submit queue
    const StaticArray waitSemaphores = {_imageAvailableSemaphores[nextFrame]};
    const StaticArray waitStages{{vk::PipelineStageFlags{
        vk::PipelineStageFlagBits::eColorAttachmentOutput}}};
    const StaticArray signalSemaphores{{_renderFinishedSemaphores[nextFrame]}};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = asserted_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount =
            asserted_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    checkSuccess(
        _device.graphicsQueue().submit(
            1, &submitInfo, _swapchain.currentFence()),
        "submit");

    const ImVec2 viewportSize = _imguiRenderer.centerAreaSize();
    const bool viewportResized =
        asserted_cast<uint32_t>(viewportSize.x) != _viewportExtent.width ||
        asserted_cast<uint32_t>(viewportSize.y) != _viewportExtent.height;
    // TODO: Queue viewport resize until imgui resizing drag is not active?
    // TODO: End gesture when mouse is released on top of imgui

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) || _window.resized() ||
        viewportResized)
        recreateSwapchainAndRelated(scopeAlloc.child_scope());
}
