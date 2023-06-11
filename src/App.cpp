#include "App.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vector_relational.hpp>
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

App::App(ScopedScratch scopeAlloc, const Settings &settings)
{
    const auto &tl = [](const char *stage, std::function<void()> const &fn)
    {
        Timer t;
        fn();
        printf("%s took %.2fs\n", stage, t.getSeconds());
    };

    tl("Window creation",
       [&]
       {
           _window = std::make_unique<Window>(
               Pair<uint32_t, uint32_t>{WIDTH, HEIGHT}, "prosper");
       });
    tl("Device creation",
       [&]
       {
           _device = std::make_unique<Device>(
               scopeAlloc.child_scope(), _window->ptr(), settings.device);
       });

    _staticDescriptorsAlloc =
        std::make_unique<DescriptorAllocator>(_generalAlloc, _device.get());

    _swapchain = std::make_unique<Swapchain>(
        _device.get(), SwapchainConfig{
                           scopeAlloc.child_scope(),
                           _device.get(),
                           {_window->width(), _window->height()}});

    _commandBuffers = allocateCommandBuffers(_device.get());

    _viewportExtent =
        _swapchain->config().extent; // This is a clang-tidy false-negative
    _resources =
        std::make_unique<RenderResources>(_generalAlloc, _device.get());

    _cam = std::make_unique<Camera>(
        scopeAlloc.child_scope(), _device.get(), _staticDescriptorsAlloc.get());
    _world = std::make_unique<World>(
        scopeAlloc.child_scope(), _device.get(), settings.scene,
        settings.deferredLoading);

    const Timer gpuPassesInitTimer;
    _lightClustering = std::make_unique<LightClustering>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout(),
        _world->_dsLayouts);
    _renderer = std::make_unique<Renderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        Renderer::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->_dsLayouts,
        });
    _gbufferRenderer = std::make_unique<GBufferRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _cam->descriptorSetLayout(), _world->_dsLayouts);
    _deferredShading = std::make_unique<DeferredShading>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(),
        DeferredShading::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->_dsLayouts,
        });
    _rtRenderer = std::make_unique<RTRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout(),
        _world->_dsLayouts);
    _skyboxRenderer = std::make_unique<SkyboxRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _world->_dsLayouts);
    _debugRenderer = std::make_unique<DebugRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout());
    _toneMap = std::make_unique<ToneMap>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get());
    _imguiRenderer = std::make_unique<ImGuiRenderer>(
        _device.get(), _resources.get(), _swapchain->config().extent,
        _window->ptr(), _swapchain->config());
    _recompileTime = std::chrono::file_clock::now();
    printf("GPU pass init took %.2fs\n", gpuPassesInitTimer.getSeconds());

    _profiler = std::make_unique<Profiler>(_generalAlloc, _device.get());

    const auto &allocs = _device->memoryAllocations();
    printf("Active GPU allocations:\n");
    printf(
        "  Buffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.buffers / 1000 / 1000));
    printf(
        "  TexelBuffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.texelBuffers / 1000 / 1000));
    printf(
        "  Images: %uMB\n",
        asserted_cast<uint32_t>(allocs.images / 1000 / 1000));

    _cam->init(_world->_scenes[_world->_currentScene].camera);
    _cam->perspective(
        _viewportExtent.width / static_cast<float>(_viewportExtent.height));

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _imageAvailableSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
        _renderFinishedSemaphores.push_back(
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{}));
    }
}

App::~App()
{
    for (auto &semaphore : _renderFinishedSemaphores)
        _device->logical().destroy(semaphore);
    for (auto &semaphore : _imageAvailableSemaphores)
        _device->logical().destroy(semaphore);
}

void App::run()
{
    LinearAllocator scopeBackingAlloc{megabytes(256)};
    while (_window->open())
    {
        _profiler->startCpuFrame();

        scopeBackingAlloc.reset();
        ScopedScratch scopeAlloc{scopeBackingAlloc};

        {
            const auto _s = _profiler->createCpuScope("Window::startFrame");
            _window->startFrame();
        }

        handleMouseGestures();

        recompileShaders(scopeAlloc.child_scope());

        drawFrame(scopeAlloc.child_scope());

        InputHandler::instance().clearSingleFrameGestures();
        _cam->clearChangedThisFrame();

        _profiler->endCpuFrame();
    }

    // Wait for in flight rendering actions to finish
    _device->logical().waitIdle();
}

void App::recreateViewportRelated()
{
    // Wait for resources to be out of use
    _device->logical().waitIdle();

    _resources->destroyResources();

    const ImVec2 viewportSize = _imguiRenderer->centerAreaSize();
    _viewportExtent = vk::Extent2D{
        asserted_cast<uint32_t>(viewportSize.x),
        asserted_cast<uint32_t>(viewportSize.y),
    };

    _cam->perspective(
        PerspectiveParameters{
            .fov = radians(CAMERA_FOV),
            .zN = CAMERA_NEAR,
            .zF = CAMERA_FAR,
        },
        _viewportExtent.width / static_cast<float>(_viewportExtent.height));
}

void App::recreateSwapchainAndRelated(ScopedScratch scopeAlloc)
{
    while (_window->width() == 0 && _window->height() == 0)
    {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device->logical().waitIdle();

    _resources->destroyResources();

    { // Drop the config as we should always use swapchain's active config
        const SwapchainConfig config{
            scopeAlloc.child_scope(),
            _device.get(),
            {_window->width(), _window->height()}};
        _swapchain->recreate(config);
    }

    _imguiRenderer->recreate(_swapchain->config().extent);
}

void App::recompileShaders(ScopedScratch scopeAlloc)
{
    const auto _s = _profiler->createCpuScope("App::recompileShaders");

    if (!_recompileShaders)
        return;

    const Timer checkTime;
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
    _device->logical().waitIdle();

    printf("Recompiling shaders\n");

    const Timer t;

    _lightClustering->recompileShaders(
        scopeAlloc.child_scope(), _cam->descriptorSetLayout(),
        _world->_dsLayouts);
    _renderer->recompileShaders(
        scopeAlloc.child_scope(),
        Renderer::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->_dsLayouts,
        });
    _gbufferRenderer->recompileShaders(
        scopeAlloc.child_scope(), _cam->descriptorSetLayout(),
        _world->_dsLayouts);
    _deferredShading->recompileShaders(
        scopeAlloc.child_scope(),
        DeferredShading::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->_dsLayouts,
        });
    _rtRenderer->recompileShaders(
        scopeAlloc.child_scope(), _cam->descriptorSetLayout(),
        _world->_dsLayouts);
    _skyboxRenderer->recompileShaders(
        scopeAlloc.child_scope(), _world->_dsLayouts);
    _debugRenderer->recompileShaders(
        scopeAlloc.child_scope(), _cam->descriptorSetLayout());
    _toneMap->recompileShaders(scopeAlloc.child_scope());

    printf("Shaders recompiled in %.2fs\n", t.getSeconds());

    _recompileTime = std::chrono::file_clock::now();
}

void App::handleMouseGestures()
{
    const auto _s = _profiler->createCpuScope("App::handleMouseGestures");

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

            const auto params = _cam->parameters();
            const auto fromTarget = params.eye - params.target;

            const auto horizontalRotatedFromTarget =
                mat3(rotate(-drag.x, params.up)) * fromTarget;

            const auto right =
                normalize(cross(horizontalRotatedFromTarget, params.up));

            const auto newFromTarget =
                mat3(rotate(drag.y, right)) * horizontalRotatedFromTarget;
            const auto flipUp =
                dot(right, cross(newFromTarget, params.up)) < 0.0;

            _cam->offset = CameraOffset{
                .eye = newFromTarget - fromTarget,
                .flipUp = flipUp,
            };
        }
        else if (gesture->type == MouseGestureType::TrackPlane)
        {
            const auto params = _cam->parameters();
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

            _cam->offset = CameraOffset{
                .eye = offset,
                .target = offset,
            };
        }
        else if (gesture->type == MouseGestureType::TrackZoom)
        {
            if (!_cam->offset.has_value())
            {
                const auto &params = _cam->parameters();

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
                    _cam->offset = offset;
                }
            }
        }
        else
            throw std::runtime_error("Unknown mouse gesture");
    }
    else
    {
        if (_cam->offset.has_value())
        {
            _cam->applyOffset();
        }
    }
}

void App::drawFrame(ScopedScratch scopeAlloc)
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const uint32_t nextFrame = asserted_cast<uint32_t>(_swapchain->nextFrame());

    const uint32_t nextImage =
        nextSwapchainImage(scopeAlloc.child_scope(), nextFrame);

    _profiler->startGpuFrame(nextFrame);

    const auto profilerDatas = _profiler->getPreviousData(scopeAlloc);

    capFramerate();

    _imguiRenderer->startFrame();

    const UiChanges uiChanges = drawUi(scopeAlloc.child_scope(), profilerDatas);

    const vk::Rect2D renderArea{
        .offset = {0, 0},
        .extent = _viewportExtent,
    };

    _world->uploadMaterialDatas(nextFrame);

    assert(
        renderArea.offset.x == 0 && renderArea.offset.y == 0 &&
        "Camera update assumes no render offset");
    _cam->updateBuffer(
        nextFrame, uvec2{renderArea.extent.width, renderArea.extent.height});

    _world->updateUniformBuffers(*_cam, nextFrame, scopeAlloc.child_scope());

    const auto &scene = _world->currentScene();

    updateDebugLines(scene, nextFrame);

    const auto cb = _commandBuffers[nextFrame];
    cb.reset();

    cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    render(
        cb, renderArea,
        RenderIndices{
            .nextFrame = nextFrame,
            .nextImage = nextImage,
        },
        scene, uiChanges);

    _world->handleDeferredLoading(
        scopeAlloc.child_scope(), cb, nextFrame, *_profiler);

    _profiler->endGpuFrame(cb);

    cb.end();

    const bool shouldResizeSwapchain = !submitAndPresent(cb, nextFrame);

    handleResizes(scopeAlloc.child_scope(), shouldResizeSwapchain);
}

uint32_t App::nextSwapchainImage(ScopedScratch scopeAlloc, uint32_t nextFrame)
{

    const vk::Semaphore imageAvailable = _imageAvailableSemaphores[nextFrame];
    Optional<uint32_t> nextImage = _swapchain->acquireNextImage(imageAvailable);
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
            _device->graphicsQueue().submit(1, &submitInfo, vk::Fence{}),
            "recreate_swap_dummy_submit");

        // Recreate the swap chain as necessary
        recreateSwapchainAndRelated(scopeAlloc.child_scope());
        nextImage = _swapchain->acquireNextImage(imageAvailable);
    }

    return *nextImage;
}

void App::capFramerate()
{
    // Enforce fps cap by spinlocking to have any hope to be somewhat consistent
    // Note that this is always based on the previous frame so it only limits
    // fps and doesn't help actual frame timing
    const float minDt =
        _useFpsLimit ? 1.f / static_cast<float>(_fpsLimit) : 0.f;
    while (_frameTimer.getSeconds() < minDt)
    {
        ;
    }
    _frameTimer.reset();
}

App::UiChanges App::drawUi(
    ScopedScratch scopeAlloc, const Array<Profiler::ScopeData> &profilerDatas)
{
    UiChanges ret;

    drawOptions();

    drawRendererSettings(ret);

    drawProfiling(scopeAlloc.child_scope(), profilerDatas);

    drawMemory();

    return ret;
}

void App::drawOptions()
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text(
        "%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
        ImGui::GetIO().Framerate);

    ImGui::Checkbox("Limit FPS", &_useFpsLimit);
    if (_useFpsLimit)
    {
        ImGui::DragInt("##FPS limit value", &_fpsLimit, 5.f, 30, 250);
    }

    ImGui::Checkbox("Recompile shaders", &_recompileShaders);

    ImGui::End();
}

void App::drawRendererSettings(UiChanges &uiChanges)
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 235.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin(
        "Renderer settings ", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // TODO: Droplist for main renderer type
    uiChanges.rtPickedThisFrame =
        ImGui::Checkbox("Render RT", &_renderRT) && _renderRT;
    if (!_renderRT)
        ImGui::Checkbox("Use deferred shading", &_renderDeferred);

    if (ImGui::CollapsingHeader("Tone Map", ImGuiTreeNodeFlags_DefaultOpen))
        _toneMap->drawUi();

    if (ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (_renderRT)
            _rtRenderer->drawUi();
        else if (_renderDeferred)
            _deferredShading->drawUi();
        else
            _renderer->drawUi();
    }

    ImGui::End();
}

void App::drawProfiling(
    ScopedScratch scopeAlloc, const Array<Profiler::ScopeData> &profilerDatas)

{

    ImGui::SetNextWindowPos(
        ImVec2{static_cast<float>(WIDTH) - 300.f, 60.f},
        ImGuiCond_FirstUseEver);
    ImGui::Begin("Profiling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    size_t longestNameLength = 0;
    for (const auto &t : profilerDatas)
        if (t.name.size() > longestNameLength)
            longestNameLength = asserted_cast<size_t>(t.name.size());

    // Double the maximum name length for headroom
    String tmp{scopeAlloc};
    tmp.resize(longestNameLength * 2);
    const auto leftJustified =
        [&tmp, longestNameLength](StrSpan str, size_t extraWidth = 0)
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
                ImGui::Text("%s %.3fms", leftJustified(t.name), t.gpuMillis);
        if (!profilerDatas.empty())
        {
            static int scopeIndex = 0;
            const char *comboTitle =
                profilerDatas[scopeIndex].gpuMillis < 0.f
                    ? "##EmptyGPUScopeTitle"
                    : profilerDatas[scopeIndex].name.data();
            if (ImGui::BeginCombo("##GPUScopeData", comboTitle, 0))
            {
                for (int n = 0; n < asserted_cast<int>(profilerDatas.size());
                     n++)
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

                ImGui::Text("Raster prim per prim: %.2f", rasterPrimPerPrim);
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
                ImGui::Text("%s %.3fms", leftJustified(t.name), t.cpuMillis);
    }

    ImGui::End();
}

void App::drawMemory()
{
    ImGui::SetNextWindowPos(
        ImVec2{
            static_cast<float>(WIDTH) - 300.f,
            static_cast<float>(HEIGHT) - 300.f},
        ImGuiCond_FirstUseEver);

    ImGui::Begin("Memory", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    const MemoryAllocationBytes &allocs = _device->memoryAllocations();
    ImGui::Text("Active GPU allocations:\n");
    ImGui::Text(
        "  Buffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.buffers / 1000 / 1000));
    ImGui::Text(
        "  TexelBuffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.texelBuffers / 1000 / 1000));
    ImGui::Text(
        "  Images: %uMB\n",
        asserted_cast<uint32_t>(allocs.images / 1000 / 1000));

    ImGui::End();
}

void App::updateDebugLines(const Scene &scene, uint32_t nextFrame)
{
    auto &debugLines = _resources->debugLines[nextFrame];
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
}

void App::render(
    vk::CommandBuffer cb, const vk::Rect2D &renderArea,
    const RenderIndices &indices, const Scene &scene,
    const UiChanges &uiChanges)
{
    const LightClustering::Output lightClusters = _lightClustering->record(
        cb, scene, *_cam, _viewportExtent, indices.nextFrame, _profiler.get());

    ImageHandle illumination;
    if (_renderRT)
    {
        illumination =
            _rtRenderer
                ->record(
                    cb, *_world, *_cam, renderArea, indices.nextFrame,
                    uiChanges.rtPickedThisFrame, _profiler.get())
                .illumination;
    }
    else
    {
        ImageHandle depth;
        // Opaque
        if (_renderDeferred)
        {
            const GBufferRenderer::Output gbuffer = _gbufferRenderer->record(
                cb, *_world, *_cam, renderArea, indices.nextFrame,
                _profiler.get());

            illumination = _deferredShading
                               ->record(
                                   cb, *_world, *_cam,
                                   DeferredShading::Input{
                                       .gbuffer = gbuffer,
                                       .lightClusters = lightClusters,
                                   },
                                   indices.nextFrame, _profiler.get())
                               .illumination;

            _resources->images.release(gbuffer.albedoRoughness);
            _resources->images.release(gbuffer.normalMetalness);

            depth = gbuffer.depth;
        }
        else
        {
            const Renderer::OpaqueOutput output = _renderer->recordOpaque(
                cb, *_world, *_cam, renderArea, lightClusters,
                indices.nextFrame, _profiler.get());
            illumination = output.illumination;
            depth = output.depth;
        }

        // Transparent
        _renderer->recordTransparent(
            cb, *_world, *_cam,
            Renderer::RecordInOut{
                .illumination = illumination,
                .depth = depth,
            },
            lightClusters, indices.nextFrame, _profiler.get());

        _skyboxRenderer->record(
            cb, *_world,
            SkyboxRenderer::RecordInOut{
                .illumination = illumination,
                .depth = depth,
            },
            indices.nextFrame, _profiler.get());

        _debugRenderer->record(
            cb, *_cam,
            DebugRenderer::RecordInOut{
                .color = illumination,
                .depth = depth,
            },
            indices.nextFrame, _profiler.get());

        _resources->images.release(depth);
    }
    _resources->images.release(lightClusters.pointers);
    _resources->texelBuffers.release(lightClusters.indicesCount);
    _resources->texelBuffers.release(lightClusters.indices);

    const ImageHandle toneMapped =
        _toneMap->record(cb, illumination, indices.nextFrame, _profiler.get())
            .toneMapped;

    _resources->images.release(illumination);

    blitToneMapped(cb, toneMapped);
    _resources->images.release(toneMapped);

    const vk::Rect2D backbufferArea{
        .offset = {0, 0},
        .extent = _swapchain->config().extent,
    };
    _imguiRenderer->endFrame(cb, backbufferArea, _profiler.get());

    blitFinalComposite(cb, indices.nextImage);
}

void App::blitToneMapped(vk::CommandBuffer cb, ImageHandle toneMapped)
{
    // Blit tonemapped into cleared final composite before drawing ui on top
    {
        const StaticArray barriers{{
            _resources->images.transitionBarrier(
                toneMapped,
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .accessMask = vk::AccessFlagBits2::eTransferRead,
                    .layout = vk::ImageLayout::eTransferSrcOptimal,
                }),
            _resources->finalComposite.transitionBarrier(ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                .accessMask = vk::AccessFlagBits2::eTransferWrite,
                .layout = vk::ImageLayout::eTransferDstOptimal,
            }),
        }};

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });
    }

    // This scope has a barrier, but that's intentional as it should contain
    // both the clear and the blit
    const auto _s = _profiler->createCpuGpuScope(cb, "BlitToneMapped");

    const vk::ClearColorValue clearColor{0.f, 0.f, 0.f, 0.f};
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    cb.clearColorImage(
        _resources->finalComposite.handle, vk::ImageLayout::eTransferDstOptimal,
        &clearColor, 1, &subresourceRange);

    // Memory barrier for finalComposite, layout is already correct
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{},
        {
            vk::MemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            },
        },
        {}, {});

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

    const ImVec2 dstOffset = _imguiRenderer->centerAreaOffset();
    const ImVec2 dstSize = _imguiRenderer->centerAreaSize();
    const vk::Extent2D backbufferExtent = _swapchain->config().extent;
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
    const vk::ImageBlit blit = {
        .srcSubresource = layers,
        .srcOffsets = srcOffsets,
        .dstSubresource = layers,
        .dstOffsets = dstOffsets,
    };
    cb.blitImage(
        _resources->images.nativeHandle(toneMapped),
        vk::ImageLayout::eTransferSrcOptimal, _resources->finalComposite.handle,
        vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
}

void App::blitFinalComposite(vk::CommandBuffer cb, uint32_t nextImage)
{
    // Blit to support different internal rendering resolution (and color
    // format?) the future

    const auto &swapImage = _swapchain->image(nextImage);

    const StaticArray barriers{{
        _resources->finalComposite.transitionBarrier(ImageState{
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
        const auto _s = _profiler->createCpuGpuScope(cb, "BlitFinalComposite");

        const vk::ImageSubresourceLayers layers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1};

        assert(
            _resources->finalComposite.extent.width == swapImage.extent.width);
        assert(
            _resources->finalComposite.extent.height ==
            swapImage.extent.height);
        const std::array offsets{
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{
                asserted_cast<int32_t>(_swapchain->config().extent.width),
                asserted_cast<int32_t>(_swapchain->config().extent.height),
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
            _resources->finalComposite.handle,
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
}

bool App::submitAndPresent(vk::CommandBuffer cb, uint32_t nextFrame)
{
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
        _device->graphicsQueue().submit(
            1, &submitInfo, _swapchain->currentFence()),
        "submit");

    return _swapchain->present(signalSemaphores);
}

void App::handleResizes(ScopedScratch scopeAlloc, bool shouldResizeSwapchain)
{
    const ImVec2 viewportSize = _imguiRenderer->centerAreaSize();
    const bool viewportResized =
        asserted_cast<uint32_t>(viewportSize.x) != _viewportExtent.width ||
        asserted_cast<uint32_t>(viewportSize.y) != _viewportExtent.height;
    // TODO: End gesture when mouse is released on top of imgui

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (shouldResizeSwapchain || _window->resized())
        recreateSwapchainAndRelated(scopeAlloc.child_scope());
    else if (viewportResized)
    { // Don't recreate viewport related on the same frame as swapchain is
      // resized since we don't know the new viewport area until the next frame
        recreateViewportRelated();
    }
}
