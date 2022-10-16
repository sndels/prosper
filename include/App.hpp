#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Camera.hpp"
#include "Device.hpp"
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

class App
{
  public:
    App(const std::filesystem::path &scene, bool enableDebugLayers);
    ~App();

    App(const App &other) = delete;
    App(App &&other) = delete;
    App &operator=(const App &other) = delete;
    App &operator=(App &&other) = delete;

    void run();

  private:
    void recompileShaders();
    void recreateSwapchainAndRelated();

    void handleMouseGestures();
    void drawFrame();

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources

    SwapchainConfig _swapConfig;
    Swapchain _swapchain;
    std::vector<vk::CommandBuffer> _swapCommandBuffers;

    // Stored here, managed by (earliest) passes that write to them
    RenderResources _resources;

    vk::DescriptorPool _descriptorPool;

    Camera _cam;
    World _world;

    LightClustering _lightClustering;
    Renderer _renderer;
    RTRenderer _rtRenderer;
    SkyboxRenderer _skyboxRenderer;
    ToneMap _toneMap;
    ImGuiRenderer _imguiRenderer;

    Profiler _profiler;

    bool _useFpsLimit{true};
    int32_t _fpsLimit{140};
    bool _recompileShaders{false};
    bool _renderRT{false};

    Timer _frameTimer;
    std::chrono::time_point<std::chrono::file_clock> _recompileTime;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;
};

#endif // PROSPER_APP_HPP
