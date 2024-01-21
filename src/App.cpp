#include "App.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/transform.hpp>
#include <imgui.h>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/string.hpp>

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/VkUtils.hpp"
#include "render/DebugRenderer.hpp"
#include "render/DeferredShading.hpp"
#include "render/ForwardRenderer.hpp"
#include "render/GBufferRenderer.hpp"
#include "render/ImGuiRenderer.hpp"
#include "render/ImageBasedLighting.hpp"
#include "render/LightClustering.hpp"
#include "render/RenderResources.hpp"
#include "render/RtReference.hpp"
#include "render/SkyboxRenderer.hpp"
#include "render/TemporalAntiAliasing.hpp"
#include "render/TextureDebug.hpp"
#include "render/ToneMap.hpp"
#include "render/dof/DepthOfField.hpp"
#include "render/rtdi/RtDirectIllumination.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "utils/InputHandler.hpp"
#include "utils/Utils.hpp"

using namespace glm;
using namespace wheels;
using namespace std::chrono_literals;

namespace
{

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;

StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> allocateCommandBuffers(
    Device *device)
{
    StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> ret;

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

App::App(const Settings &settings)
: _generalAlloc{megabytes(16)}
, _fileChangePollingAlloc{megabytes(1)}
{
    LinearAllocator scratchBacking{megabytes(16)};
    ScopedScratch scopeAlloc{scratchBacking};

    const auto &tl = [](const char *stage, std::function<void()> const &fn)
    {
        const Timer t;
        fn();
        printf("%s took %.2fs\n", stage, t.getSeconds());
    };

    tl("Window creation",
       [&]
       {
           _window = std::make_unique<Window>(
               Pair<uint32_t, uint32_t>{WIDTH, HEIGHT}, "prosper",
               &_inputHandler);
       });
    tl("Device creation",
       [&]
       {
           _device = std::make_unique<Device>(
               _generalAlloc, scopeAlloc.child_scope(), _window->ptr(),
               settings.device);
       });

    _staticDescriptorsAlloc =
        std::make_unique<DescriptorAllocator>(_generalAlloc, _device.get());

    _swapchain = std::make_unique<Swapchain>(
        _device.get(), SwapchainConfig{
                           scopeAlloc.child_scope(),
                           _device.get(),
                           {_window->width(), _window->height()}});

    _commandBuffers = allocateCommandBuffers(_device.get());

    // We don't know the extent in member inits
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    _viewportExtent = _swapchain->config().extent;

    _resources =
        std::make_unique<RenderResources>(_generalAlloc, _device.get());

    _cam = std::make_unique<Camera>(
        scopeAlloc.child_scope(), _device.get(), &_resources->constantsRing,
        _staticDescriptorsAlloc.get());
    _world = std::make_unique<World>(
        _generalAlloc, scopeAlloc.child_scope(), _device.get(),
        &_resources->constantsRing, settings.scene);

    const Timer gpuPassesInitTimer;
    _lightClustering = std::make_unique<LightClustering>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _forwardRenderer = std::make_unique<ForwardRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        ForwardRenderer::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->dsLayouts(),
        });
    _gbufferRenderer = std::make_unique<GBufferRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _cam->descriptorSetLayout(), _world->dsLayouts());
    _deferredShading = std::make_unique<DeferredShading>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(),
        DeferredShading::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->dsLayouts(),
        });
    _rtDirectIllumination = std::make_unique<RtDirectIllumination>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _rtReference = std::make_unique<RtReference>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _skyboxRenderer = std::make_unique<SkyboxRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _cam->descriptorSetLayout(), _world->dsLayouts());
    _debugRenderer = std::make_unique<DebugRenderer>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout());
    _toneMap = std::make_unique<ToneMap>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get());
    _imguiRenderer = std::make_unique<ImGuiRenderer>(
        _device.get(), _resources.get(), _swapchain->config().extent,
        _window->ptr(), _swapchain->config());
    _textureDebug = std::make_unique<TextureDebug>(
        _generalAlloc, scopeAlloc.child_scope(), _device.get(),
        _resources.get(), _staticDescriptorsAlloc.get());
    _depthOfField = std::make_unique<DepthOfField>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout());
    _imageBasedLighting = std::make_unique<ImageBasedLighting>(
        scopeAlloc.child_scope(), _device.get(), _staticDescriptorsAlloc.get());
    _temporalAntiAliasing = std::make_unique<TemporalAntiAliasing>(
        scopeAlloc.child_scope(), _device.get(), _resources.get(),
        _staticDescriptorsAlloc.get(), _cam->descriptorSetLayout());
    _recompileTime = std::chrono::file_clock::now();
    printf("GPU pass init took %.2fs\n", gpuPassesInitTimer.getSeconds());

    _profiler = std::make_unique<Profiler>(_generalAlloc, _device.get());

    _cam->init(_sceneCameraTransform, _cameraParameters);
    _cam->updateResolution(
        uvec2{_viewportExtent.width, _viewportExtent.height});

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _imageAvailableSemaphores[i] =
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{});
        _renderFinishedSemaphores[i] =
            _device->logical().createSemaphore(vk::SemaphoreCreateInfo{});
    }
    _ctorScratchHighWatermark = asserted_cast<uint32_t>(
        scratchBacking.allocated_byte_count_high_watermark());
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
    LinearAllocator scopeBackingAlloc{megabytes(16)};
    Timer updateDelta;
    _lastTimeChange = std::chrono::high_resolution_clock::now();
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
        handleKeyboardInput(updateDelta.getSeconds());
        updateDelta.reset();

        recompileShaders(scopeAlloc.child_scope());

        _resources->startFrame();
        _world->startFrame();

        drawFrame(
            scopeAlloc.child_scope(),
            asserted_cast<uint32_t>(
                scopeBackingAlloc.allocated_byte_count_high_watermark()));

        _inputHandler.clearSingleFrameGestures();
        _cam->endFrame();

        _world->endFrame();

        _profiler->endCpuFrame();
    }

    // Wait for in flight rendering actions to finish
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    _device->graphicsQueue().waitIdle();
}

void App::recreateViewportRelated()
{
    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    _device->graphicsQueue().waitIdle();

    _resources->destroyResources();

    if (_drawUi)
    {
        const ImVec2 viewportSize = _imguiRenderer->centerAreaSize();
        _viewportExtent = vk::Extent2D{
            asserted_cast<uint32_t>(viewportSize.x),
            asserted_cast<uint32_t>(viewportSize.y),
        };
    }
    else
        _viewportExtent = _swapchain->config().extent;

    _cam->updateResolution(
        uvec2{_viewportExtent.width, _viewportExtent.height});
}

void App::recreateSwapchainAndRelated(ScopedScratch scopeAlloc)
{
    while (_window->width() == 0 && _window->height() == 0)
    {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    _device->graphicsQueue().waitIdle();

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
    {
        if (_fileChanges.valid())
            // This blocks until the future is done which should be fine as it
            // causes at most one frame drop.
            _fileChanges = {};
        return;
    }

    if (!_fileChanges.valid())
    {
        // Push a new async task that polls files to avoid holding back
        // rendering if it lags.
        _fileChanges = std::async(
            std::launch::async,
            [this]()
            {
                const Timer checkTime;
                auto shadersIterator =
                    std::filesystem::recursive_directory_iterator(
                        resPath("shader"));
                const uint32_t shaderFileBound = 128;
                HashSet<std::filesystem::path> changedFiles{
                    _fileChangePollingAlloc, shaderFileBound};
                for (const auto &entry : shadersIterator)
                {
                    if (entry.last_write_time() > _recompileTime)
                        changedFiles.insert(entry.path().lexically_normal());
                }
                WHEELS_ASSERT(changedFiles.capacity() == shaderFileBound);

                if (checkTime.getSeconds() > 0.01f)
                    fprintf(
                        stderr, "Shader timestamp check is laggy: %.1fms\n",
                        checkTime.getSeconds() * 1000.f);

                return changedFiles;
            });
        return;
    }

    const std::future_status status = _fileChanges.wait_for(0s);
    WHEELS_ASSERT(
        status != std::future_status::deferred &&
        "The future should never be lazy");
    if (status == std::future_status::timeout)
        return;

    const HashSet<std::filesystem::path> changedFiles{
        WHEELS_MOV(_fileChanges.get())};
    if (changedFiles.empty())
        return;

    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    _device->graphicsQueue().waitIdle();

    // We might get here before the changed shaders are retouched completely,
    // e.g. if clang-format takes a bit. Let's try to be safe with an extra
    // wait to avoid reading them mid-write.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    printf("Recompiling shaders\n");

    const Timer t;

    _lightClustering->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _forwardRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        ForwardRenderer::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->dsLayouts(),
        });
    _gbufferRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _deferredShading->recompileShaders(
        scopeAlloc.child_scope(), changedFiles,
        DeferredShading::InputDSLayouts{
            .camera = _cam->descriptorSetLayout(),
            .lightClusters = _lightClustering->descriptorSetLayout(),
            .world = _world->dsLayouts(),
        });
    _rtDirectIllumination->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _rtReference->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _skyboxRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout(),
        _world->dsLayouts());
    _debugRenderer->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout());
    _toneMap->recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _textureDebug->recompileShaders(scopeAlloc.child_scope(), changedFiles);
    _depthOfField->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout());
    _imageBasedLighting->recompileShaders(
        scopeAlloc.child_scope(), changedFiles);
    _temporalAntiAliasing->recompileShaders(
        scopeAlloc.child_scope(), changedFiles, _cam->descriptorSetLayout());

    printf("Shaders recompiled in %.2fs\n", t.getSeconds());

    _recompileTime = std::chrono::file_clock::now();
}

void App::handleMouseGestures()
{
    const auto _s = _profiler->createCpuScope("App::handleMouseGestures");

    // Gestures adapted from Max Liani
    // https://maxliani.wordpress.com/2021/06/08/offline-to-realtime-camera-manipulation/

    const auto &gesture = _inputHandler.mouseGesture();
    if (gesture.has_value() && _camFreeLook)
    {
        if (gesture->type == MouseGestureType::TrackBall)
        {

            const auto dragScale = 1.f / 400.f;
            const auto drag =
                (gesture->currentPos - gesture->startPos) * dragScale;

            const auto transform = _cam->transform();
            const auto fromTarget = transform.eye - transform.target;

            const auto horizontalRotatedFromTarget =
                mat3(rotate(-drag.x, transform.up)) * fromTarget;

            const auto right =
                normalize(cross(horizontalRotatedFromTarget, transform.up));

            const auto newFromTarget =
                mat3(rotate(drag.y, right)) * horizontalRotatedFromTarget;
            const auto flipUp =
                dot(right, cross(newFromTarget, transform.up)) < 0.0;

            _cam->gestureOffset = CameraOffset{
                .eye = newFromTarget - fromTarget,
                .flipUp = flipUp,
            };
        }
        else if (gesture->type == MouseGestureType::TrackPlane)
        {
            const auto transform = _cam->transform();
            const auto from_target = transform.eye - transform.target;
            const auto dist_target = length(from_target);

            // TODO: Adjust for aspect ratio difference between film and window
            const auto drag_scale = [&]
            {
                const auto params = _cam->parameters();
                auto tanHalfFov = tan(params.fov * 0.5f);
                return dist_target * tanHalfFov /
                       (static_cast<float>(_viewportExtent.height) * 0.5f);
            }();
            const auto drag =
                (gesture->currentPos - gesture->startPos) * drag_scale;

            const auto right = normalize(cross(from_target, transform.up));
            const auto cam_up = normalize(cross(right, from_target));

            const auto offset = right * (drag.x) + cam_up * (drag.y);

            _cam->gestureOffset = CameraOffset{
                .eye = offset,
                .target = offset,
            };
        }
        else if (gesture->type == MouseGestureType::TrackZoom)
        {
            if (!_cam->gestureOffset.has_value())
            {
                const auto &transform = _cam->transform();

                const auto to_target = transform.target - transform.eye;
                const auto dist_target = length(to_target);
                const auto fwd = to_target / dist_target;

                const auto scroll_scale = dist_target * 0.1f;

                const auto offset = CameraOffset{
                    .eye = fwd * gesture->verticalScroll * scroll_scale,
                };

                // Make sure we don't get too close to get stuck
                const auto offsettransform = transform.apply(offset);
                if (all(greaterThan(
                        abs(offsettransform.eye - transform.target),
                        vec3(compMax(
                            0.01f *
                            max(offsettransform.eye, transform.target))))))
                {
                    _cam->gestureOffset = offset;
                }
            }
        }
        else
            throw std::runtime_error("Unknown mouse gesture");
    }
    else
    {
        if (_cam->gestureOffset.has_value())
        {
            _cam->applyGestureOffset();
        }
    }
}

void App::handleKeyboardInput(float deltaS)
{
    const StaticArray<KeyState, KeyCount> &keyStates = _inputHandler.keyboard();

    if (keyStates[KeyI] == KeyState::Pressed)
    {
        _drawUi = !_drawUi;
        _forceViewportRecreate = true;
    }

    if (_camFreeLook)
    {
        const float baseSpeed = 2.f;
        vec3 speed{0.f};

        if (keyStates[KeyW] == KeyState::Pressed ||
            keyStates[KeyW] == KeyState::Held)
            speed.z += baseSpeed;
        if (keyStates[KeyS] == KeyState::Pressed ||
            keyStates[KeyS] == KeyState::Held)
            speed.z -= baseSpeed;
        if (keyStates[KeyD] == KeyState::Pressed ||
            keyStates[KeyD] == KeyState::Held)
            speed.x += baseSpeed;
        if (keyStates[KeyA] == KeyState::Pressed ||
            keyStates[KeyA] == KeyState::Held)
            speed.x -= baseSpeed;
        if (keyStates[KeyE] == KeyState::Pressed ||
            keyStates[KeyE] == KeyState::Held)
            speed.y += baseSpeed;
        if (keyStates[KeyQ] == KeyState::Pressed ||
            keyStates[KeyQ] == KeyState::Held)
            speed.y -= baseSpeed;

        if (keyStates[KeyShift] == KeyState::Held)
            speed *= 2.f;
        if (keyStates[KeyCtrl] == KeyState::Held)
            speed *= 0.5f;

        speed *= deltaS;

        if (length(speed) > 0.f)
        {
            const CameraTransform &transform = _cam->transform();
            const Optional<CameraOffset> &offset = _cam->gestureOffset;

            const vec3 eye = offset.has_value() ? transform.eye + offset->eye
                                                : transform.eye;
            const vec3 target = offset.has_value()
                                    ? transform.target + offset->target
                                    : transform.target;

            const vec3 fwd = normalize(target - eye);
            const vec3 right = normalize(cross(fwd, transform.up));
            const vec3 up = normalize(cross(right, fwd));

            const vec3 movement =
                right * speed.x + fwd * speed.z + up * speed.y;

            _cam->applyOffset(CameraOffset{
                .eye = movement,
                .target = movement,
            });
        }
    }
}

void App::drawFrame(ScopedScratch scopeAlloc, uint32_t scopeHighWatermark)
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const uint32_t nextFrame = asserted_cast<uint32_t>(_swapchain->nextFrame());

    const uint32_t nextImage =
        nextSwapchainImage(scopeAlloc.child_scope(), nextFrame);

    _profiler->startGpuFrame(nextFrame);

    const auto profilerDatas = _profiler->getPreviousData(scopeAlloc);

    capFramerate();

    UiChanges uiChanges;
    if (_drawUi)
    {
        _imguiRenderer->startFrame(_profiler.get());

        uiChanges =
            drawUi(scopeAlloc.child_scope(), profilerDatas, scopeHighWatermark);
    }
    // Clear for new frame after UI was drawn
    _sceneStats = SceneStats{};

    const vk::Rect2D renderArea{
        .offset = {0, 0},
        .extent = _viewportExtent,
    };

    const float timeS = currentTimelineTimeS();
    _world->updateAnimations(timeS, _profiler.get());

    _world->updateScene(
        scopeAlloc.child_scope(), &_sceneCameraTransform, &_sceneStats,
        _profiler.get());

    _world->uploadMeshDatas(scopeAlloc.child_scope(), nextFrame);

    // -1 seems like a safe value here since an 8 sample halton sequence is
    // used. See A Survey of Temporal Antialiasing Techniques by Yang, Liu and
    // Salvi for details.
    const float lodBias = _applyTaa ? -1.f : 0.f;
    _world->uploadMaterialDatas(nextFrame, lodBias);

    if (_isPlaying || _forceCamUpdate || uiChanges.timeTweaked)
    {
        _cam->lookAt(_sceneCameraTransform);

        const CameraParameters &params = _world->currentCamera();
        // This makes sure we copy the new params over when a camera is
        // changed, or for the first camera
        _cameraParameters = params;
        _cam->setParameters(params);
        _forceCamUpdate = false;
        // Disable free look for animated cameras
        _camFreeLook = !_world->isCurrentCameraDynamic();
    }

    WHEELS_ASSERT(
        renderArea.offset.x == 0 && renderArea.offset.y == 0 &&
        "Camera update assumes no render offset");
    _cam->updateBuffer();

    {
        auto _s = _profiler->createCpuScope("World::updateBuffers");
        _world->updateBuffers(scopeAlloc.child_scope());
    }

    updateDebugLines(_world->currentScene(), nextFrame);

    const auto cb = _commandBuffers[nextFrame];
    cb.reset();

    cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    if (_applyIbl && !_imageBasedLighting->isGenerated())
        _imageBasedLighting->recordGeneration(
            scopeAlloc.child_scope(), cb, *_world, nextFrame, _profiler.get());

    render(
        scopeAlloc.child_scope(), cb, renderArea,
        RenderIndices{
            .nextFrame = nextFrame,
            .nextImage = nextImage,
        },
        uiChanges);

    _newSceneDataLoaded = _world->handleDeferredLoading(cb, *_profiler);

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
        const vk::PipelineStageFlags dstMask{};
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &imageAvailable,
            // Fun fact: Windows NV and Linux AMD-PRO don't seem to use this
            // or have special cases for null
            .pWaitDstStageMask = &dstMask,
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

float App::currentTimelineTimeS() const
{
    if (!_isPlaying)
        return _timeOffsetS;

    const auto now = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> dt = now - _lastTimeChange;
    const float deltaS = dt.count();

    const float timeS = deltaS + _timeOffsetS;
    return timeS;
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
    ScopedScratch scopeAlloc, const Array<Profiler::ScopeData> &profilerDatas,
    uint32_t scopeHighWatermark)
{
    const auto _s = _profiler->createCpuScope("App::drawUi");

    UiChanges ret;
    // Actual scene change happens after the frame so let's initialize here with
    // last frame's value
    // TODO:
    // At least _newSceneDataLoaded would probably be less surprising if
    // combined to the dirty flag somewhere else
    ret.rtDirty = _sceneChanged || _newSceneDataLoaded;

    _sceneChanged = _world->drawSceneUi();

    ret.rtDirty |= drawCameraUi();

    drawOptions();

    drawRendererSettings(ret);

    drawProfiling(scopeAlloc.child_scope(), profilerDatas);

    drawMemory(scopeHighWatermark);

    drawSceneStats();

    ret.rtDirty |= _isPlaying;
    ret.timeTweaked |= drawTimeline();
    ret.rtDirty |= ret.timeTweaked;

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
        // Drag doesn't clamp values that are input as text
        _fpsLimit = std::max(_fpsLimit, 30);
    }

    ImGui::Checkbox("Recompile shaders", &_recompileShaders);

    if (ImGui::Checkbox("Texture Debug", &_textureDebugActive) &&
        !_textureDebugActive)
        _resources->images.clearDebug();

    ImGui::End();
}

void App::drawRendererSettings(UiChanges &uiChanges)
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 235.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin(
        "Renderer settings ", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // TODO: Droplist for main renderer type
    uiChanges.rtDirty |=
        ImGui::Checkbox("Reference RT", &_referenceRt) && _referenceRt;
    uiChanges.rtDirty |= ImGui::Checkbox("Depth of field (WIP)", &_renderDoF);
    ImGui::Checkbox("Temporal Anti-Aliasing", &_applyTaa);

    if (!_referenceRt)
    {
        ImGui::Checkbox("Deferred shading", &_renderDeferred);

        if (_renderDeferred)
            uiChanges.rtDirty =
                ImGui::Checkbox("RT direct illumination", &_deferredRt);
    }

    if (!_applyTaa)
        _cam->setJitter(false);
    else
    {
        if (ImGui::CollapsingHeader(
                "Temporal Anti-Aliasing", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Jitter", &_applyJitter);
            _cam->setJitter(_applyJitter);
            _temporalAntiAliasing->drawUi();
        }
    }

    if (ImGui::CollapsingHeader("Tone Map", ImGuiTreeNodeFlags_DefaultOpen))
        _toneMap->drawUi();

    if (ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (_referenceRt)
            _rtReference->drawUi();
        else
        {
            if (_renderDeferred)
            {
                if (_deferredRt)
                    _rtDirectIllumination->drawUi();
                else
                    _deferredShading->drawUi();
            }
            else
                _forwardRenderer->drawUi();
        }
        uiChanges.rtDirty |= ImGui::Checkbox("IBL", &_applyIbl);
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
        WHEELS_ASSERT(longestNameLength + extraWidth <= tmp.size());

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
                    ? "Pipeline stats##EmptyGPUScopeTitle"
                    : profilerDatas[scopeIndex].name.data();
            if (ImGui::BeginCombo("##GPUScopeData", comboTitle, 0))
            {
                const int scopeCount = asserted_cast<int>(profilerDatas.size());
                for (int n = 0; n < scopeCount; n++)
                {
                    const Profiler::ScopeData &scopeData = profilerDatas[n];
                    // Only have scopes that have gpu stats
                    if (scopeData.gpuStats.has_value())
                    {
                        const bool selected = scopeIndex == n;
                        if (ImGui::Selectable(scopeData.name.data(), selected))
                            scopeIndex = n;

                        if (selected)
                            ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (profilerDatas[scopeIndex].gpuStats.has_value())
            {
                const auto &stats = *profilerDatas[scopeIndex].gpuStats;
                const auto &swapExtent = _viewportExtent;
                const uint32_t pixelCount =
                    swapExtent.width * swapExtent.height;

                // Stats from AMD's 'D3D12 Right On Queue' GDC2016
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

void App::drawMemory(uint32_t scopeHighWatermark)
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

    TlsfAllocator::Stats const &allocStats = _generalAlloc.stats();

    ImGui::Text("High watermarks:\n");
    ImGui::Text(
        "  ctors : %uKB\n",
        asserted_cast<uint32_t>(_ctorScratchHighWatermark) / 1000);
    ImGui::Text(
        "  deferred general: %uMB\n",
        asserted_cast<uint32_t>(
            _world->deferredLoadingGeneralAllocatorHighWatermark() / 1000 /
            1000));
    ImGui::Text(
        "  world: %uKB\n",
        asserted_cast<uint32_t>(_world->linearAllocatorHighWatermark() / 1000));
    ImGui::Text(
        "  general: %uKB\n",
        asserted_cast<uint32_t>(
            allocStats.allocated_byte_count_high_watermark / 1000));
    ImGui::Text("  frame scope: %uKB\n", scopeHighWatermark / 1000);

    ImGui::Text("General allocator stats:\n");

    ImGui::Text(
        "  count: %u\n", asserted_cast<uint32_t>(allocStats.allocation_count));
    ImGui::Text(
        "  small count: %u\n",
        asserted_cast<uint32_t>(allocStats.small_allocation_count));
    ImGui::Text(
        "  size: %uKB\n",
        asserted_cast<uint32_t>(allocStats.allocated_byte_count / 1000));
    ImGui::Text(
        "  free size: %uKB\n",
        asserted_cast<uint32_t>(allocStats.free_byte_count / 1000));

    ImGui::End();
}

bool App::drawTimeline()
{
    bool timeTweaked = false;
    const Scene &scene = _world->currentScene();
    if (scene.endTimeS > 0.f)
    {
        ImGui::SetNextWindowPos(ImVec2{400, 50}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2{400, 50}, ImGuiCond_Appearing);
        ImGui::Begin(
            "Animation timeline", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        // Fit slider to the window
        const ImVec2 region = ImGui::GetContentRegionAvail();
        ImGui::PushItemWidth(region.x);

        float timeS = currentTimelineTimeS();
        if (ImGui::SliderFloat(
                "##TimelineTime", &timeS, 0.f, scene.endTimeS, "%.3fs"))
        {
            _lastTimeChange = std::chrono::high_resolution_clock::now();
            _timeOffsetS = timeS;
            _timeOffsetS = std::clamp(timeS, 0.f, scene.endTimeS);
            timeTweaked = true;
        }

        ImGui::PopItemWidth();

        if (currentTimelineTimeS() > scene.endTimeS)
        {
            _lastTimeChange = std::chrono::high_resolution_clock::now();
            _timeOffsetS = 0.f;
            timeTweaked = true;
        }

        const float buttonWidth = 30.f;
        if (ImGui::Button("|<", ImVec2(buttonWidth, 0)))
        {
            _lastTimeChange = std::chrono::high_resolution_clock::now();
            _timeOffsetS = 0;
            timeTweaked = true;
        }

        ImGui::SameLine();
        if (_isPlaying)
        {
            if (ImGui::Button("||", ImVec2(buttonWidth, 0)))
            {
                _isPlaying = false;
                _lastTimeChange = std::chrono::high_resolution_clock::now();
                _timeOffsetS = timeS;
                timeTweaked = true;
            }
        }
        else if (ImGui::Button(">", ImVec2(buttonWidth, 0)))
        {
            _isPlaying = true;
            _lastTimeChange = std::chrono::high_resolution_clock::now();
            timeTweaked = true;
        }

        ImGui::SameLine();
        if (ImGui::Button(">|", ImVec2(buttonWidth, 0)))
        {
            _lastTimeChange = std::chrono::high_resolution_clock::now();
            _timeOffsetS = scene.endTimeS;
            timeTweaked = true;
        }

        ImGui::End();
    }

    return timeTweaked;
}

bool App::drawCameraUi()
{
    bool changed = false;

    ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    _forceCamUpdate |= _world->drawCameraUi();
    if (_world->isCurrentCameraDynamic())
        ImGui::Checkbox("Free look", &_camFreeLook);

    CameraParameters params = _cam->parameters();

    if (_camFreeLook)
    {
        // TODO: Tweak this in millimeters?
        changed |= ImGui::DragFloat(
            "Aperture Diameter", &params.apertureDiameter, 0.00001f, 0.0000001f,
            0.1f, "%.6f");
        changed |= ImGui::DragFloat(
            "FocusDistance", &params.focusDistance, 0.01f, 0.001f, 100.f);

        float fovDegrees = degrees(params.fov);
        if (ImGui::DragFloat("Field of View", &fovDegrees, 0.1f, 0.1f, 179.f))
        {
            params.fov = radians(fovDegrees);
            changed = true;
        }

        // Set before drawing focal length as this updates it after fov changes
        // TODO: Just use focal length instead of fov as the only parameter?
        if (changed)
            _cam->setParameters(params);
    }
    else
    {
        ImGui::Text("Aperture Diameter: %.6f", params.apertureDiameter);
        ImGui::Text("FocusDistance: %.3f", params.focusDistance);

        const float fovDegrees = degrees(params.fov);
        ImGui::Text("Field of View: %.3f", fovDegrees);
    }

    ImGui::Text("Focal length: %.3fmm", params.focalLength * 1e3);

    ImGui::End();

    return changed;
}

void App::drawSceneStats() const
{
    ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Total triangles: %u", _sceneStats.totalTriangleCount);
    ImGui::Text("Total meshlets: %u", _sceneStats.totalMeshletCount);
    ImGui::Text("Total meshes: %u", _sceneStats.totalMeshCount);
    ImGui::Text("Total nodes: %u", _sceneStats.totalNodeCount);
    ImGui::Text("Animated nodes: %u", _sceneStats.animatedNodeCount);

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
    ScopedScratch scopeAlloc, vk::CommandBuffer cb,
    const vk::Rect2D &renderArea, const RenderIndices &indices,
    const UiChanges &uiChanges)
{
    bool blasesAdded = false;
    if (_referenceRt || _rtDirectIllumination)
    {
        auto _s = _profiler->createCpuGpuScope(cb, "BuildTLAS");
        blasesAdded = _world->buildAccelerationStructures(cb);
    }

    const LightClusteringOutput lightClusters = _lightClustering->record(
        scopeAlloc.child_scope(), cb, *_world, *_cam, _viewportExtent,
        indices.nextFrame, _profiler.get());

    ImageHandle illumination;
    if (_referenceRt)
    {
        _rtDirectIllumination->releasePreserved();
        _temporalAntiAliasing->releasePreserved();

        illumination =
            _rtReference
                ->record(
                    scopeAlloc.child_scope(), cb, *_world, *_cam, renderArea,
                    RtReference::Options{
                        .depthOfField = _renderDoF,
                        .ibl = _applyIbl,
                        .colorDirty = uiChanges.rtDirty || blasesAdded,
                    },
                    indices.nextFrame, _profiler.get())
                .illumination;
    }
    else
    {
        // Need to clean up after toggling rt off to not "leak" the resources
        _rtReference->releasePreserved();

        ImageHandle velocity;
        ImageHandle depth;
        // Opaque
        if (_renderDeferred)
        {
            const GBufferRendererOutput gbuffer = _gbufferRenderer->record(
                scopeAlloc.child_scope(), cb, *_world, *_cam, renderArea,
                indices.nextFrame, &_sceneStats, _profiler.get());

            if (_deferredRt)
                illumination =
                    _rtDirectIllumination
                        ->record(
                            scopeAlloc.child_scope(), cb, *_world, *_cam,
                            gbuffer, uiChanges.rtDirty || blasesAdded,
                            indices.nextFrame, _profiler.get())
                        .illumination;
            else
            {
                _rtDirectIllumination->releasePreserved();

                illumination =
                    _deferredShading
                        ->record(
                            scopeAlloc.child_scope(), cb, *_world, *_cam,
                            DeferredShading::Input{
                                .gbuffer = gbuffer,
                                .lightClusters = lightClusters,
                            },
                            indices.nextFrame, _applyIbl, _profiler.get())
                        .illumination;
            }

            _resources->images.release(gbuffer.albedoRoughness);
            _resources->images.release(gbuffer.normalMetalness);

            velocity = gbuffer.velocity;
            depth = gbuffer.depth;
        }
        else
        {
            _rtDirectIllumination->releasePreserved();

            const ForwardRenderer::OpaqueOutput output =
                _forwardRenderer->recordOpaque(
                    scopeAlloc.child_scope(), cb, *_world, *_cam, renderArea,
                    lightClusters, indices.nextFrame, _applyIbl, &_sceneStats,
                    _profiler.get());
            illumination = output.illumination;
            velocity = output.velocity;
            depth = output.depth;
        }

        _skyboxRenderer->record(
            scopeAlloc.child_scope(), cb, *_world, *_cam,
            SkyboxRenderer::RecordInOut{
                .illumination = illumination,
                .velocity = velocity,
                .depth = depth,
            },
            _profiler.get());

        // Transparent
        _forwardRenderer->recordTransparent(
            scopeAlloc.child_scope(), cb, *_world, *_cam,
            ForwardRenderer::TransparentInOut{
                .illumination = illumination,
                .depth = depth,
            },
            lightClusters, indices.nextFrame, &_sceneStats, _profiler.get());

        _debugRenderer->record(
            scopeAlloc.child_scope(), cb, *_cam,
            DebugRenderer::RecordInOut{
                .color = illumination,
                .depth = depth,
            },
            indices.nextFrame, _profiler.get());

        if (_applyTaa)
        {
            const TemporalAntiAliasing::Output taaOutput =
                _temporalAntiAliasing->record(
                    scopeAlloc.child_scope(), cb, *_cam,
                    TemporalAntiAliasing::Input{
                        .illumination = illumination,
                        .velocity = velocity,
                        .depth = depth,
                    },
                    indices.nextFrame, _profiler.get());

            _resources->images.release(illumination);
            illumination = taaOutput.resolvedIllumination;
        }
        else
            _temporalAntiAliasing->releasePreserved();

        // TODO:
        // Do DoF on raw illumination and have a separate stabilizing TAA pass
        // that doesn't blend foreground/background (Karis/Abadie).
        if (_renderDoF)
        {
            const DepthOfField::Output dofOutput = _depthOfField->record(
                scopeAlloc.child_scope(), cb, *_cam,
                DepthOfField::Input{
                    .illumination = illumination,
                    .depth = depth,
                },
                indices.nextFrame, _profiler.get());

            _resources->images.release(illumination);
            illumination = dofOutput.combinedIlluminationDoF;
        }

        _resources->images.release(velocity);
        _resources->images.release(depth);
    }
    _resources->images.release(lightClusters.pointers);
    _resources->texelBuffers.release(lightClusters.indicesCount);
    _resources->texelBuffers.release(lightClusters.indices);

    const ImageHandle toneMapped =
        _toneMap
            ->record(
                scopeAlloc.child_scope(), cb, illumination, indices.nextFrame,
                _profiler.get())
            .toneMapped;

    _resources->images.release(illumination);

    if (_textureDebugActive)
    {
        const ImageHandle debugOutput = _textureDebug->record(
            scopeAlloc.child_scope(), cb, renderArea.extent, indices.nextFrame,
            _profiler.get());

        blitColorToFinalComposite(cb, debugOutput);

        _resources->images.release(debugOutput);
    }
    else
        blitColorToFinalComposite(cb, toneMapped);

    _resources->images.release(toneMapped);

    if (_drawUi)
    {
        _world->drawDeferredLoadingUi();

        if (_textureDebugActive)
            // Draw this after so that the first frame debug is active for a new
            // texture, we draw black instead of a potentially wrong output from
            // the shared texture that wasn't protected yet
            _textureDebug->drawUi();

        const vk::Rect2D backbufferArea{
            .offset = {0, 0},
            .extent = _swapchain->config().extent,
        };
        _imguiRenderer->endFrame(cb, backbufferArea, _profiler.get());
    }

    blitFinalComposite(cb, indices.nextImage);
}

void App::blitColorToFinalComposite(
    vk::CommandBuffer cb, ImageHandle toneMapped)
{
    // Blit tonemapped into cleared final composite before drawing ui on top
    {
        const StaticArray barriers{{
            *_resources->images.transitionBarrier(
                toneMapped, ImageState::TransferSrc, true),
            *_resources->finalComposite.transitionBarrier(
                ImageState::TransferDst, true),
        }};

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });
    }

    // This scope has a barrier, but that's intentional as it should contain
    // both the clear and the blit
    const auto _s =
        _profiler->createCpuGpuScope(cb, "blitColorToFinalComposite");

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

    const vk::Extent2D backbufferExtent = _swapchain->config().extent;
    ivec2 dstOffset;
    ivec2 dstSize;
    if (_drawUi)
    {
        const ImVec2 offset = _imguiRenderer->centerAreaOffset();
        const ImVec2 size = _imguiRenderer->centerAreaSize();
        dstOffset = ivec2{static_cast<int32_t>(offset.x), offset.y};
        dstSize = ivec2{size.x, size.y};
    }
    else
    {
        dstOffset = ivec2{0, 0};
        dstSize = ivec2{
            asserted_cast<int32_t>(backbufferExtent.width),
            asserted_cast<int32_t>(backbufferExtent.height),
        };
    }

    const std::array dstOffsets{
        vk::Offset3D{
            std::min(
                dstOffset.x,
                asserted_cast<int32_t>(backbufferExtent.width - 1)),
            std::min(
                dstOffset.y,
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
        *_resources->finalComposite.transitionBarrier(
            ImageState::TransferSrc, true),
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

        WHEELS_ASSERT(
            _resources->finalComposite.extent.width == swapImage.extent.width);
        WHEELS_ASSERT(
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
    const StaticArray waitSemaphores{_imageAvailableSemaphores[nextFrame]};
    const StaticArray waitStages{vk::PipelineStageFlags{
        vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    const StaticArray signalSemaphores{_renderFinishedSemaphores[nextFrame]};
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
    else if (viewportResized || _forceViewportRecreate)
    { // Don't recreate viewport related on the same frame as swapchain is
      // resized since we don't know the new viewport area until the next frame
        recreateViewportRelated();
        _forceViewportRecreate = false;
    }
}
