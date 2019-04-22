#include "App.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>

#include "Constants.hpp"
#include "InputHandler.hpp"
#include "Vertex.hpp"

using namespace glm;

using std::cout;
using std::cerr;
using std::endl;

namespace {
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    const float CAMERA_FOV = 59.f;
    const float CAMERA_NEAR = 0.001f;
    const float CAMERA_FAR = 512.f;
}

App::~App()
{
    _device.logical().destroy(_vkDescriptorPool);
}

void App::init()
{
    _window.init(WIDTH, HEIGHT, "prosper");
    _device.init(_window.ptr());

    const SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(), _window.height()}
    );

    // Resources tied to specific swap images via command buffers
    createDescriptorPool(swapConfig.imageCount);

    _cam.createUniformBuffers(&_device, swapConfig.imageCount);
    _cam.createDescriptorSets(
        _vkDescriptorPool,
        swapConfig.imageCount,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
    );

    _world.loadGLTF(
        &_device,
        swapConfig.imageCount,
        resPath("glTF/FlightHelmet/glTF/FlightHelmet.gltf")
    );

    _cam.lookAt(
        vec3{0.25f, 0.2f, 0.75f},
        vec3{0.f},
        vec3{0.f, 1.f, 0.f}
    );

    _renderer.init(&_device);
    recreateSwapchainAndRelated();
}

void App::run() 
{
    while (_window.open()) {
        _window.startFrame();

        const auto& mouse = InputHandler::instance().mouse();
        if (mouse.leftDown && mouse.currentPos != mouse.lastPos) {
            _cam.orbit(
                mouse.currentPos,
                mouse.lastPos,
                vec2(_window.width(), _window.height()) / 2.f
            );
        }
        if (mouse.rightDown && mouse.currentPos != mouse.lastPos) {
            _cam.scaleOrbit(
                mouse.currentPos.y,
                 mouse.lastPos.y,
                _window.height() / 2.f
            );
        }
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    _device.logical().waitIdle();
}

void App::recreateSwapchainAndRelated()
{
    while (_window.width() == 0 && _window.height() == 0) {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device.logical().waitIdle();

    _renderer.destroySwapchainRelated();
    _swapchain.destroy();

    const SwapchainConfig swapConfig = selectSwapchainConfig(
        &_device,
        {_window.width(),
        _window.height()}
    );

    _renderer.createSwapchainRelated(
        swapConfig,
        _cam.descriptorSetLayout(),
        _world._dsLayouts
    );
    _swapchain.create(&_device, swapConfig);

    _cam.perspective(
        radians(CAMERA_FOV),
        _window.width() / static_cast<float>(_window.height()),
        CAMERA_NEAR,
        CAMERA_FAR
    );
}

void App::createDescriptorPool(const uint32_t swapImageCount)
{
    const vk::DescriptorPoolSize poolSize{
            vk::DescriptorType::eUniformBuffer,
            swapImageCount // descriptorCount, camera and mesh instances
        };
    _vkDescriptorPool = _device.logical().createDescriptorPool({
        {}, // flags
        poolSize.descriptorCount, // max sets
        1,
        &poolSize
    });
}

void App::drawFrame()
{
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t nextFrame = _swapchain.nextFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&]{
        const auto imageAvailable = _renderer.imageAvailable(nextFrame);
        auto nextImage = _swapchain.acquireNextImage(imageAvailable);
        while (!nextImage.has_value()) {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(imageAvailable);
        }

        return nextImage.value();
    }();

    const auto signalSemaphores = _renderer.drawFrame(_world, _cam, _swapchain, nextImage);

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) ||
        _window.resized())
        recreateSwapchainAndRelated();

}
