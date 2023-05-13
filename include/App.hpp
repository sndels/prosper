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
    void recreateViewportRelated();
    void createCommandBuffers();

    void handleMouseGestures();
    void drawFrame(wheels::ScopedScratch scopeAlloc);

    wheels::CstdlibAllocator _generalAlloc;

    std::unique_ptr<Window>
        _window; // Needs to be valid before and after everything else
    std::unique_ptr<Device>
        _device; // Needs to be valid before and after all other vk resources

    std::unique_ptr<Swapchain> _swapchain;
    wheels::StaticArray<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT>
        _commandBuffers;

    vk::Extent2D _viewportExtent{};

    // Stored here, managed by (earliest) passes that write to them
    std::unique_ptr<RenderResources> _resources;

    std::unique_ptr<Camera> _cam;
    std::unique_ptr<World> _world;

    std::unique_ptr<LightClustering> _lightClustering;
    std::unique_ptr<Renderer> _renderer;
    std::unique_ptr<GBufferRenderer> _gbufferRenderer;
    std::unique_ptr<DeferredShading> _deferredShading;
    std::unique_ptr<RTRenderer> _rtRenderer;
    std::unique_ptr<SkyboxRenderer> _skyboxRenderer;
    std::unique_ptr<DebugRenderer> _debugRenderer;
    std::unique_ptr<ToneMap> _toneMap;
    std::unique_ptr<ImGuiRenderer> _imguiRenderer;

    std::unique_ptr<Profiler> _profiler;

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
