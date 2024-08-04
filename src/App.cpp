#include "App.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif // GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/transform.hpp>
#include <imgui.h>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/string.hpp>

#include "Allocators.hpp"
#include "gfx/DescriptorAllocator.hpp"
#include "gfx/VkUtils.hpp"
#include "render/RenderResources.hpp"
#include "render/Renderer.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "utils/InputHandler.hpp"
#include "utils/Logger.hpp"
#include "utils/Ui.hpp"
#include "utils/Utils.hpp"

using namespace glm;
using namespace wheels;
using namespace std::chrono_literals;

namespace
{

StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> allocateCommandBuffers()
{
    StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> ret;

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = gDevice.graphicsPool(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    checkSuccess(
        gDevice.logical().allocateCommandBuffers(&allocInfo, ret.data()),
        "Failed to allocate command buffers");

    return ret;
}

} // namespace

App::App(std::filesystem::path scenePath) noexcept
: m_fileChangePollingAlloc{megabytes(1)}
, m_scenePath{WHEELS_MOV(scenePath)}
, m_swapchain{OwningPtr<Swapchain>{gAllocators.general}}
, m_cam{OwningPtr<Camera>{gAllocators.general}}
, m_world{OwningPtr<World>{gAllocators.general}}
, m_renderer{OwningPtr<Renderer>{gAllocators.general}}
{
}

App::~App()
{
    for (auto &semaphore : m_renderFinishedSemaphores)
    {
        if (semaphore)
            gDevice.logical().destroy(semaphore);
    }
    for (auto &semaphore : m_imageAvailableSemaphores)
    {
        if (semaphore)
            gDevice.logical().destroy(semaphore);
    }
}

void App::init(ScopedScratch scopeAlloc)
{
    {
        const SwapchainConfig &config = SwapchainConfig{
            scopeAlloc.child_scope(), {gWindow.width(), gWindow.height()}};
        m_swapchain->init(config);
    }

    m_commandBuffers = allocateCommandBuffers();

    // We don't know the extent in member inits
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_viewportExtent = m_swapchain->config().extent;

    m_constantsRing.init(
        vk::BufferUsageFlagBits::eStorageBuffer,
        asserted_cast<uint32_t>(kilobytes(16)), "ConstantsRing");

    m_cam->init(scopeAlloc.child_scope(), &m_constantsRing);

    m_world->init(scopeAlloc.child_scope(), &m_constantsRing, m_scenePath);

    m_renderer->init(
        scopeAlloc.child_scope(), m_swapchain->config(),
        m_cam->descriptorSetLayout(), m_world->dsLayouts());

    m_recompileTime = std::chrono::file_clock::now();

    m_cam->lookAt(m_sceneCameraTransform);
    m_cam->setParameters(m_cameraParameters);
    m_cam->updateResolution(
        uvec2{m_viewportExtent.width, m_viewportExtent.height});

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        m_imageAvailableSemaphores[i] =
            gDevice.logical().createSemaphore(vk::SemaphoreCreateInfo{});
        m_renderFinishedSemaphores[i] =
            gDevice.logical().createSemaphore(vk::SemaphoreCreateInfo{});
    }
}

void App::setInitScratchHighWatermark(size_t value)
{
    m_ctorScratchHighWatermark = asserted_cast<uint32_t>(value);
}

void App::run()
{
    LinearAllocator scopeBackingAlloc{megabytes(16)};
    Timer updateDelta;
    m_lastTimeChange = std::chrono::high_resolution_clock::now();

    try
    {
        while (gWindow.open())
        {
            gProfiler.startCpuFrame();

            scopeBackingAlloc.reset();
            ScopedScratch scopeAlloc{scopeBackingAlloc};

            {
                PROFILER_CPU_SCOPE("Window::startFrame");
                gWindow.startFrame();
            }

            // The cursor position callback doesn't get called on every frame
            // that has cursor movement if the cursor is hidden so let's poll
            // the position
            gWindow.pollCursorPosition();
            handleMouseGestures();
            handleKeyboardInput(updateDelta.getSeconds());
            updateDelta.reset();

            recompileShaders(scopeAlloc.child_scope());

            m_constantsRing.startFrame();

            m_world->startFrame();

            m_renderer->startFrame();

            drawFrame(
                scopeAlloc.child_scope(),
                asserted_cast<uint32_t>(
                    scopeBackingAlloc.allocated_byte_count_high_watermark()));

            gInputHandler.clearSingleFrameGestures();
            m_cam->endFrame();

            m_world->endFrame();

            gProfiler.endCpuFrame();
        }
    }
    catch (std::exception &)
    {
        // Wait for in flight rendering actions to finish to make app cleanup
        // valid. Don't wait for device idle as async loading might be using the
        // transfer queue simultaneously
        gDevice.graphicsQueue().waitIdle();
        throw;
    }
    LOG_INFO("Closing window");

    // Wait for in flight rendering actions to finish
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    gDevice.graphicsQueue().waitIdle();
}

void App::recreateViewportRelated()
{
    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    gDevice.graphicsQueue().waitIdle();

    m_renderer->recreateViewportRelated();
    m_viewportExtent = m_drawUi ? m_renderer->viewportExtentInUi()
                                : m_swapchain->config().extent;

    m_cam->updateResolution(
        uvec2{m_viewportExtent.width, m_viewportExtent.height});
}

void App::recreateSwapchainAndRelated(ScopedScratch scopeAlloc)
{
    while (gWindow.width() == 0 && gWindow.height() == 0)
    {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    gDevice.graphicsQueue().waitIdle();

    {
        const SwapchainConfig config{
            scopeAlloc.child_scope(), {gWindow.width(), gWindow.height()}};
        m_swapchain->recreate(config);
    }
}

void App::recompileShaders(ScopedScratch scopeAlloc)
{
    PROFILER_CPU_SCOPE("App::recompileShaders");

    if (!m_recompileShaders)
    {
        if (m_fileChanges.valid())
            // This blocks until the future is done which should be fine as it
            // causes at most one frame drop.
            m_fileChanges = {};
        return;
    }

    if (!m_fileChanges.valid())
    {
        // Push a new async task that polls files to avoid holding back
        // rendering if it lags.
        m_fileChanges = std::async(
            std::launch::async,
            [this]()
            {
                const Timer checkTime;
                auto shadersIterator =
                    std::filesystem::recursive_directory_iterator(
                        resPath("shader"));
                const uint32_t shaderFileBound = 128;
                HashSet<std::filesystem::path> changedFiles{
                    m_fileChangePollingAlloc, shaderFileBound};
                for (const auto &entry : shadersIterator)
                {
                    if (entry.last_write_time() > m_recompileTime)
                        changedFiles.insert(entry.path().lexically_normal());
                }
                WHEELS_ASSERT(changedFiles.capacity() == shaderFileBound);

                if (checkTime.getSeconds() > 0.2f)
                    LOG_WARN(
                        "Shader timestamp check is laggy: %.1fms",
                        checkTime.getSeconds() * 1000.f);

                return changedFiles;
            });
        return;
    }

    const std::future_status status = m_fileChanges.wait_for(0s);
    WHEELS_ASSERT(
        status != std::future_status::deferred &&
        "The future should never be lazy");
    if (status == std::future_status::timeout)
        return;

    const HashSet<std::filesystem::path> changedFiles{
        WHEELS_MOV(m_fileChanges.get())};
    if (changedFiles.empty())
        return;

    // Wait for resources to be out of use
    // Don't wait for device idle as async loading might be using the transfer
    // queue simultaneously
    gDevice.graphicsQueue().waitIdle();

    // We might get here before the changed shaders are retouched completely,
    // e.g. if clang-format takes a bit. Let's try to be safe with an extra
    // wait to avoid reading them mid-write.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    m_renderer->recompileShaders(
        scopeAlloc.child_scope(), m_cam->descriptorSetLayout(),
        m_world->dsLayouts(), changedFiles);

    m_recompileTime = std::chrono::file_clock::now();
}

void App::handleMouseGestures()
{
    PROFILER_CPU_SCOPE("App::handleMouseGestures");

    // Gestures adapted from Max Liani
    // https://maxliani.wordpress.com/2021/06/08/offline-to-realtime-camera-manipulation/

    const auto &gesture = gInputHandler.mouseGesture();
    if (gesture.has_value() && m_camFreeLook)
    {
        if (gesture->type == MouseGestureType::TrackBall)
        {

            const auto dragScale = 1.f / 400.f;
            const auto drag =
                (gesture->currentPos - gesture->startPos) * dragScale;

            const auto transform = m_cam->transform();
            const auto fromTarget = transform.eye - transform.target;

            const auto horizontalRotatedFromTarget =
                mat3(rotate(-drag.x, transform.up)) * fromTarget;

            const auto right =
                normalize(cross(horizontalRotatedFromTarget, transform.up));

            const auto newFromTarget =
                mat3(rotate(drag.y, right)) * horizontalRotatedFromTarget;
            const auto flipUp =
                dot(right, cross(newFromTarget, transform.up)) < 0.0;

            m_cam->gestureOffset = CameraOffset{
                .eye = newFromTarget - fromTarget,
                .flipUp = flipUp,
            };
        }
        else if (gesture->type == MouseGestureType::TrackPlane)
        {
            const auto transform = m_cam->transform();
            const auto from_target = transform.eye - transform.target;
            const auto dist_target = length(from_target);

            // TODO: Adjust for aspect ratio difference between film and window
            const auto drag_scale = [&]
            {
                const auto params = m_cam->parameters();
                auto tanHalfFov = tan(params.fov * 0.5f);
                return dist_target * tanHalfFov /
                       (static_cast<float>(m_viewportExtent.height) * 0.5f);
            }();
            const auto drag =
                (gesture->currentPos - gesture->startPos) * drag_scale;

            const auto right = normalize(cross(from_target, transform.up));
            const auto cam_up = normalize(cross(right, from_target));

            const auto offset = right * (drag.x) + cam_up * (drag.y);

            m_cam->gestureOffset = CameraOffset{
                .eye = offset,
                .target = offset,
            };
        }
        else if (gesture->type == MouseGestureType::TrackZoom)
        {
            if (!m_cam->gestureOffset.has_value())
            {
                const auto &transform = m_cam->transform();

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
                    m_cam->gestureOffset = offset;
                }
            }
        }
        else if (gesture->type == MouseGestureType::SelectPoint)
        {
            // Reference RT doesn't write a depth buffer so can't use the
            // texture readback
            if (m_renderer->depthAvailable() && !m_waitFocusDistance)
                m_pickFocusDistance = true;
        }
        else
            throw std::runtime_error("Unknown mouse gesture");
    }
    else
    {
        if (m_cam->gestureOffset.has_value())
        {
            m_cam->applyGestureOffset();
        }
    }
}

void App::handleKeyboardInput(float deltaS)
{
    const StaticArray<KeyState, KeyCount> &keyStates = gInputHandler.keyboard();

    if (keyStates[KeyI] == KeyState::Pressed)
    {
        m_drawUi = !m_drawUi;
        m_forceViewportRecreate = true;
    }

    if (m_camFreeLook)
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
            const CameraTransform &transform = m_cam->transform();
            const Optional<CameraOffset> &offset = m_cam->gestureOffset;

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

            m_cam->applyOffset(CameraOffset{
                .eye = movement,
                .target = movement,
            });
        }
    }
}

void App::drawFrame(ScopedScratch scopeAlloc, uint32_t scopeHighWatermark)
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const uint32_t nextFrame =
        asserted_cast<uint32_t>(m_swapchain->nextFrame());

    const uint32_t nextImage =
        nextSwapchainImage(scopeAlloc.child_scope(), nextFrame);

    gProfiler.startGpuFrame(nextFrame);

    const auto profilerDatas = gProfiler.getPreviousData(scopeAlloc);

    capFramerate();

    UiChanges uiChanges;
    if (m_drawUi)
        uiChanges = drawUi(
            scopeAlloc.child_scope(), nextFrame, profilerDatas,
            scopeHighWatermark);

    const vk::Rect2D renderArea{
        .offset = {0, 0},
        .extent = m_viewportExtent,
    };

    const float timeS = currentTimelineTimeS();
    m_world->updateAnimations(timeS);

    m_world->updateScene(
        scopeAlloc.child_scope(), &m_sceneCameraTransform,
        &m_sceneStats[nextFrame]);

    m_world->uploadMeshDatas(scopeAlloc.child_scope(), nextFrame);

    const float lodBias = m_renderer->lodBias();
    m_world->uploadMaterialDatas(nextFrame, lodBias);

    if (m_isPlaying || m_forceCamUpdate || uiChanges.timeTweaked)
    {
        // Don't needlessly reset free look movement
        // TODO:
        // Add a button to reset non-animated camera to scene defined position?
        if (m_forceCamUpdate || !m_camFreeLook)
        {
            m_cam->lookAt(m_sceneCameraTransform);

            const CameraParameters &params = m_world->currentCamera();
            // This makes sure we copy the new params over when a camera is
            // changed, or for the first camera
            m_cameraParameters = params;
            m_cam->setParameters(params);
            if (m_forceCamUpdate)
                // Disable free look for animated cameras when the camera
                // changed
                m_camFreeLook = !m_world->isCurrentCameraDynamic();
            m_forceCamUpdate = false;
        }
    }

    WHEELS_ASSERT(
        renderArea.offset.x == 0 && renderArea.offset.y == 0 &&
        "Camera update assumes no render offset");
    m_cam->updateBuffer(m_debugFrustum);

    {
        PROFILER_CPU_SCOPE("World::updateBuffers");
        m_world->updateBuffers(scopeAlloc.child_scope());
    }

    updateDebugLines(m_world->currentScene(), nextFrame);

    const auto cb = m_commandBuffers[nextFrame];
    cb.reset();

    cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    if (m_renderer->rtInUse() || m_world->unbuiltBlases())
    {
        PROFILER_CPU_GPU_SCOPE(cb, "BuildTLAS");
        uiChanges.rtDirty |=
            m_world->buildAccelerationStructures(scopeAlloc.child_scope(), cb);
    }
    Renderer::Options renderOptions{
        .rtDirty = uiChanges.rtDirty,
        .drawUi = m_drawUi,
    };
    if (m_pickFocusDistance)
    {
        const Optional<MouseGesture> &gesture = gInputHandler.mouseGesture();
        WHEELS_ASSERT(gesture.has_value());

        const vec2 offset = m_renderer->viewportOffsetInUi();
        m_pickedFocusPx = gesture->currentPos - offset;
        renderOptions.readbackDepthPx = m_pickedFocusPx;

        m_pickFocusDistance = false;
        m_waitFocusDistance = true;
    }
    const SwapchainImage swapImage = m_swapchain->image(nextImage);
    m_renderer->render(
        scopeAlloc.child_scope(), cb, *m_cam, *m_world, renderArea, swapImage,
        nextFrame, renderOptions);

    m_newSceneDataLoaded = m_world->handleDeferredLoading(cb);

    gProfiler.endGpuFrame(cb);

    cb.end();

    if (m_waitFocusDistance)
    {
        const Optional<vec4> nonLinearDepth = m_renderer->tryDepthReadback();
        if (nonLinearDepth.has_value())
        {
            const vec2 uv =
                (m_pickedFocusPx + 0.5f) /
                vec2{m_viewportExtent.width, m_viewportExtent.height};
            const vec2 clipXy = uv * 2.f - 1.f;
            const vec4 projected =
                m_cam->clipToCamera() * vec4{clipXy, nonLinearDepth->x, 1.f};
            vec3 projectedDir = vec3{projected} / projected.w;
            const float projectedDepth = length(projectedDir);
            projectedDir /= projectedDepth;

            const float cosTheta = dot(vec3{0.f, 0.f, -1.f}, projectedDir);

            CameraParameters params = m_cam->parameters();
            params.focusDistance = projectedDepth * cosTheta;
            m_cam->setParameters(params);

            m_pickedFocusPx = vec2{-1.f, -1.f};
            m_waitFocusDistance = false;
        }
    }

    const bool shouldResizeSwapchain = !submitAndPresent(cb, nextFrame);

    handleResizes(scopeAlloc.child_scope(), shouldResizeSwapchain);
}

uint32_t App::nextSwapchainImage(ScopedScratch scopeAlloc, uint32_t nextFrame)
{
    const vk::Semaphore imageAvailable = m_imageAvailableSemaphores[nextFrame];
    Optional<uint32_t> nextImage =
        m_swapchain->acquireNextImage(imageAvailable);
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
            gDevice.graphicsQueue().submit(1, &submitInfo, vk::Fence{}),
            "recreate_swap_dummy_submit");

        recreateSwapchainAndRelated(scopeAlloc.child_scope());
        nextImage = m_swapchain->acquireNextImage(imageAvailable);
    }

    return *nextImage;
}

float App::currentTimelineTimeS() const
{
    if (!m_isPlaying)
        return m_timeOffsetS;

    const auto now = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> dt = now - m_lastTimeChange;
    const float deltaS = dt.count();

    const float timeS = deltaS + m_timeOffsetS;
    return timeS;
}

void App::capFramerate()
{
    // Note that this is always based on the previous frame so it only limits
    // fps and doesn't help actual frame timing
    const float minDt =
        m_useFpsLimit ? 1.f / static_cast<float>(m_fpsLimit) : 0.f;
    while (m_frameTimer.getSeconds() < minDt)
    {
        ;
    }
    m_frameTimer.reset();
}

App::UiChanges App::drawUi(
    ScopedScratch scopeAlloc, uint32_t nextFrame,
    const Array<Profiler::ScopeData> &profilerDatas,
    uint32_t scopeHighWatermark)
{
    PROFILER_CPU_SCOPE("App::drawUi");

    UiChanges ret;
    // Actual scene change happens after the frame so let's initialize here with
    // last frame's value
    // TODO:
    // At least m_newSceneDataLoaded would probably be less surprising if
    // combined to the dirty flag somewhere else
    ret.rtDirty = m_sceneChanged || m_newSceneDataLoaded;

    m_sceneChanged = m_world->drawSceneUi();

    ret.rtDirty |= drawCameraUi();

    drawOptions();

    ret.rtDirty |= m_renderer->drawUi(*m_cam);

    drawProfiling(scopeAlloc.child_scope(), profilerDatas);

    drawMemory(scopeHighWatermark);

    drawSceneStats(nextFrame);

    ret.rtDirty |= m_isPlaying;
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

    ImGui::Checkbox("Limit FPS", &m_useFpsLimit);
    if (m_useFpsLimit)
    {
        ImGui::DragInt("##FPS limit value", &m_fpsLimit, 5.f, 30, 250);
        // Drag doesn't clamp values that are input as text
        m_fpsLimit = std::max(m_fpsLimit, 30);
    }

    ImGui::Checkbox("Recompile shaders", &m_recompileShaders);

    ImGui::End();
}

void App::drawProfiling(
    ScopedScratch scopeAlloc, const Array<Profiler::ScopeData> &profilerDatas)

{
    ImGui::SetNextWindowPos(ImVec2{600.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Profiling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    size_t longestNameLength = 0;
    for (const auto &t : profilerDatas)
        if (t.name.size() > longestNameLength)
            longestNameLength = asserted_cast<size_t>(t.name.size());

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
                const auto &swapExtent = m_viewportExtent;
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

void App::drawMemory(uint32_t scopeHighWatermark) const
{
    ImGui::SetNextWindowPos(ImVec2{1600.f, 600.f}, ImGuiCond_FirstUseEver);

    ImGui::Begin("Memory", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    const MemoryAllocationBytes &allocs = gDevice.memoryAllocations();
    ImGui::Text("Active GPU allocations:\n");
    ImGui::Text(
        "  Buffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.buffers / 1024 / 1024));
    ImGui::Text(
        "  TexelBuffers: %uMB\n",
        asserted_cast<uint32_t>(allocs.texelBuffers / 1024 / 1024));
    ImGui::Text(
        "  Images: %uMB\n",
        asserted_cast<uint32_t>(allocs.images / 1024 / 1024));

    TlsfAllocator::Stats const &allocStats = gAllocators.general.stats();

    ImGui::Text("High watermarks:\n");
    ImGui::Text(
        "  ctors : %uKB\n",
        asserted_cast<uint32_t>(m_ctorScratchHighWatermark) / 1024);
    ImGui::Text(
        "  deferred general: %uMB\n",
        asserted_cast<uint32_t>(
            gAllocators.loadingWorkerHighWatermark / 1024 / 1024));
    ImGui::Text(
        "  world: %uKB\n",
        asserted_cast<uint32_t>(
            gAllocators.world.allocated_byte_count_high_watermark() / 1024));
    ImGui::Text(
        "  general: %uMB\n",
        asserted_cast<uint32_t>(
            allocStats.allocated_byte_count_high_watermark / 1024 / 1024));
    ImGui::Text("  frame scope: %uKB\n", scopeHighWatermark / 1024);

    ImGui::Text("General allocator stats:\n");

    ImGui::Text(
        "  count: %u\n", asserted_cast<uint32_t>(allocStats.allocation_count));
    ImGui::Text(
        "  small count: %u\n",
        asserted_cast<uint32_t>(allocStats.small_allocation_count));
    ImGui::Text(
        "  size: %uKB\n",
        asserted_cast<uint32_t>(allocStats.allocated_byte_count / 1024));
    ImGui::Text(
        "  free size: %uKB\n",
        asserted_cast<uint32_t>(allocStats.free_byte_count / 1024));

    ImGui::End();
}

bool App::drawTimeline()
{
    bool timeTweaked = false;
    const Scene &scene = m_world->currentScene();
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
            m_lastTimeChange = std::chrono::high_resolution_clock::now();
            m_timeOffsetS = timeS;
            m_timeOffsetS = std::clamp(timeS, 0.f, scene.endTimeS);
            timeTweaked = true;
        }

        ImGui::PopItemWidth();

        if (currentTimelineTimeS() > scene.endTimeS)
        {
            m_lastTimeChange = std::chrono::high_resolution_clock::now();
            m_timeOffsetS = 0.f;
            timeTweaked = true;
        }

        const float buttonWidth = 30.f;
        if (ImGui::Button("|<", ImVec2(buttonWidth, 0)))
        {
            m_lastTimeChange = std::chrono::high_resolution_clock::now();
            m_timeOffsetS = 0;
            timeTweaked = true;
        }

        ImGui::SameLine();
        if (m_isPlaying)
        {
            if (ImGui::Button("||", ImVec2(buttonWidth, 0)))
            {
                m_isPlaying = false;
                m_lastTimeChange = std::chrono::high_resolution_clock::now();
                m_timeOffsetS = timeS;
                timeTweaked = true;
            }
        }
        else if (ImGui::Button(">", ImVec2(buttonWidth, 0)))
        {
            m_isPlaying = true;
            m_lastTimeChange = std::chrono::high_resolution_clock::now();
            timeTweaked = true;
        }

        ImGui::SameLine();
        if (ImGui::Button(">|", ImVec2(buttonWidth, 0)))
        {
            m_lastTimeChange = std::chrono::high_resolution_clock::now();
            m_timeOffsetS = scene.endTimeS;
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

    m_forceCamUpdate |= m_world->drawCameraUi();
    if (m_world->isCurrentCameraDynamic())
        ImGui::Checkbox("Free look", &m_camFreeLook);

    CameraParameters params = m_cam->parameters();

    if (m_camFreeLook)
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
            m_cam->setParameters(params);
    }
    else
    {
        ImGui::Text("Aperture Diameter: %.6f", params.apertureDiameter);
        ImGui::Text("FocusDistance: %.3f", params.focusDistance);

        const float fovDegrees = degrees(params.fov);
        ImGui::Text("Field of View: %.3f", fovDegrees);
    }

    ImGui::Text("Focal length: %.3fmm", params.focalLength * 1e3);

    if (m_debugFrustum.has_value())
    {
        if (ImGui::Button("Clear debug frustum"))
            m_debugFrustum.reset();

        ImGui::ColorEdit3(
            "Frustum color", &m_frustumDebugColor[0],
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
    }
    else
    {
        if (ImGui::Button("Freeze debug frustum"))
            m_debugFrustum = m_cam->getFrustumCorners();
    }

    ImGui::End();

    return changed;
}

void App::drawSceneStats(uint32_t nextFrame)
{
    const DrawStats &drawStats = m_renderer->drawStats(nextFrame);

    ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Total triangles: %u", drawStats.totalTriangleCount);
    ImGui::Text("Rasterized triangles: %u", drawStats.rasterizedTriangleCount);
    ImGui::Text("Total meshlets: %u", drawStats.totalMeshletCount);
    ImGui::Text("Drawn meshlets: %u", drawStats.drawnMeshletCount);
    ImGui::Text("Total meshes: %u", drawStats.totalMeshCount);
    ImGui::Text("Total nodes: %u", m_sceneStats[nextFrame].totalNodeCount);
    ImGui::Text(
        "Animated nodes: %u", m_sceneStats[nextFrame].animatedNodeCount);

    ImGui::End();
}

void App::updateDebugLines(const Scene &scene, uint32_t nextFrame)
{
    auto &debugLines = gRenderResources.debugLines[nextFrame];
    debugLines.reset();
    {
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

    if (m_debugFrustum.has_value())
    {
        const FrustumCorners &corners = *m_debugFrustum;

        // Near plane
        debugLines.addLine(
            corners.bottomLeftNear, corners.topLeftNear, m_frustumDebugColor);
        debugLines.addLine(
            corners.bottomLeftNear, corners.bottomRightNear,
            m_frustumDebugColor);
        debugLines.addLine(
            corners.topRightNear, corners.topLeftNear, m_frustumDebugColor);
        debugLines.addLine(
            corners.topRightNear, corners.bottomRightNear, m_frustumDebugColor);

        // Far plane
        debugLines.addLine(
            corners.bottomLeftFar, corners.topLeftFar, m_frustumDebugColor);
        debugLines.addLine(
            corners.bottomLeftFar, corners.bottomRightFar, m_frustumDebugColor);
        debugLines.addLine(
            corners.topRightFar, corners.topLeftFar, m_frustumDebugColor);
        debugLines.addLine(
            corners.topRightFar, corners.bottomRightFar, m_frustumDebugColor);

        // Pyramid edges
        debugLines.addLine(
            corners.bottomLeftNear, corners.bottomLeftFar, m_frustumDebugColor);
        debugLines.addLine(
            corners.bottomRightNear, corners.bottomRightFar,
            m_frustumDebugColor);
        debugLines.addLine(
            corners.topLeftNear, corners.topLeftFar, m_frustumDebugColor);
        debugLines.addLine(
            corners.topRightNear, corners.topRightFar, m_frustumDebugColor);
    }
}

bool App::submitAndPresent(vk::CommandBuffer cb, uint32_t nextFrame)
{
    const StaticArray waitSemaphores{m_imageAvailableSemaphores[nextFrame]};
    const StaticArray waitStages{vk::PipelineStageFlags{
        vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    const StaticArray signalSemaphores{m_renderFinishedSemaphores[nextFrame]};
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
        gDevice.graphicsQueue().submit(
            1, &submitInfo, m_swapchain->currentFence()),
        "submit");

    return m_swapchain->present(signalSemaphores);
}

void App::handleResizes(ScopedScratch scopeAlloc, bool shouldResizeSwapchain)
{
    const bool viewportResized = m_renderer->viewportResized();

    if (shouldResizeSwapchain || gWindow.resized())
        recreateSwapchainAndRelated(scopeAlloc.child_scope());
    else if (viewportResized || m_forceViewportRecreate)
    {
        // Don't recreate viewport related on the same frame as swapchain is
        // resized since we don't know the new viewport area until the next
        // frame
        recreateViewportRelated();
        m_forceViewportRecreate = false;
    }
}
