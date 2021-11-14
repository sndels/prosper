#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Camera.hpp"
#include "Device.hpp"
#include "ImGuiRenderer.hpp"
#include "Renderer.hpp"
#include "Swapchain.hpp"
#include "Timer.hpp"
#include "Window.hpp"
#include "World.hpp"

class App {
  public:
    App();
    ~App();

    App(const App &other) = delete;
    App &operator=(const App &other) = delete;

    void run();

  private:
    void recreateSwapchainAndRelated();

    void drawFrame();

    Window _window; // Needs to be valid before and after everything else
    std::shared_ptr<Device> _device =
        nullptr; // Needs to be valid before and after all other vk resources
    SwapchainConfig _swapConfig;
    ImGuiRenderer _imguiRenderer;
    vk::DescriptorPool _descriptorPool;
    Camera _cam;
    World _world;
    Swapchain _swapchain;
    std::vector<vk::CommandBuffer> _swapCommandBuffers;
    Renderer _renderer;

    bool _useFpsLimit = true;
    int32_t _fpsLimit = 60;

    Timer _frameTimer;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;
};

#endif // PROSPER_APP_HPP
