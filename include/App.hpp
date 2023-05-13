#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Camera.hpp"
#include "DebugRenderer.hpp"
#include "DeferredShading.hpp"
#include "Device.hpp"
#include "GBufferRenderer.hpp"
#include "ImGuiRenderer.hpp"
#include "LightClustering.hpp"
#include "Profiler.hpp"
#include "RTRenderer.hpp"
#include "RenderResources.hpp"
#include "Renderer.hpp"
#include "SkyboxRenderer.hpp"
#include "Swapchain.hpp"
#include "Timer.hpp"
#include "ToneMap.hpp"
#include "Window.hpp"
#include "World.hpp"

#include <filesystem>
#include <wheels/allocators/cstdlib_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/static_array.hpp>

class App
{
  public:
    App(wheels::ScopedScratch scopeAlloc, const std::filesystem::path &scene,
        bool enableDebugLayers);
    ~App();

    App(const App &other) = delete;
    App(App &&other) = delete;
    App &operator=(const App &other) = delete;
    App &operator=(App &&other) = delete;

    void run();

  private:
    void recompileShaders(wheels::ScopedScratch scopeALloc);
    void recreateSwapchainAndRelated(wheels::ScopedScratch scopeAlloc);
    void recreateViewportRelated(wheels::ScopedScratch scopeAlloc);
    void createCommandBuffers();

    void handleMouseGestures();
    void drawFrame(wheels::ScopedScratch scopeAlloc);

    wheels::CstdlibAllocator _generalAlloc;

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources

    Swapchain _swapchain;
    wheels::StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT>
        _commandBuffers;

    vk::Extent2D _viewportExtent{};

    // Stored here, managed by (earliest) passes that write to them
    RenderResources _resources;

    Camera _cam;
    World _world;

    Timer _gpuPassesInitTimer;
    LightClustering _lightClustering;
    Renderer _renderer;
    GBufferRenderer _gbufferRenderer;
    DeferredShading _deferredShading;
    RTRenderer _rtRenderer;
    SkyboxRenderer _skyboxRenderer;
    DebugRenderer _debugRenderer;
    ToneMap _toneMap;
    ImGuiRenderer _imguiRenderer;

    Profiler _profiler;

    bool _useFpsLimit{true};
    int32_t _fpsLimit{140};
    bool _recompileShaders{false};
    bool _renderRT{false};
    bool _renderDeferred{false};

    Timer _frameTimer;
    std::chrono::time_point<std::chrono::file_clock> _recompileTime;

    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        _imageAvailableSemaphores;
    wheels::StaticArray<vk::Semaphore, MAX_FRAMES_IN_FLIGHT>
        _renderFinishedSemaphores;
};

#endif // PROSPER_APP_HPP
