#ifndef PROSPER_APP_HPP
#define PROSPER_APP_HPP

#include "Camera.hpp"
#include "Device.hpp"
#include "Renderer.hpp"
#include "Swapchain.hpp"
#include "Window.hpp"
#include "World.hpp"

class App {
public:
    App();
    ~App();

    App(const App& other) = delete;
    App& operator=(const App& other) = delete;

    void run();

private:
    void recreateSwapchainAndRelated();

    void drawFrame();

    Window _window; // Needs to be valid before and after everything else
    std::shared_ptr<Device> _device = nullptr; // Needs to be valid before and after all other vk resources
    SwapchainConfig _swapConfig;
    vk::DescriptorPool _descriptorPool;
    Camera _cam;
    World _world;
    Swapchain _swapchain;
    Renderer _renderer;

};

#endif // PROSPER_APP_HPP
