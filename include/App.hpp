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
    App() = default;
    ~App();

    App(const App& other) = delete;
    App& operator=(const App& other) = delete;

    void init();
    void run();

private:
    void recreateSwapchainAndRelated();

    void createDescriptorPool(const uint32_t swapImageCount);

    void drawFrame();

    Window _window; // Needs to be valid before and after everything else
    Device _device; // Needs to be valid before and after all other vk resources
    Swapchain _swapchain;
    World _world;
    Camera _cam;
    Renderer _renderer;

    vk::DescriptorPool _vkDescriptorPool;

};

#endif // PROSPER_APP_HPP
