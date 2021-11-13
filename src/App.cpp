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

using std::cerr;
using std::cout;
using std::endl;

namespace {
const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const float CAMERA_FOV = 59.f;
const float CAMERA_NEAR = 0.001f;
const float CAMERA_FAR = 512.f;

vk::DescriptorPool createDescriptorPool(const std::shared_ptr<Device> device,
                                        const uint32_t swapImageCount) {
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount =
            swapImageCount // descriptorCount, camera and mesh instances
    };
    return device->logical().createDescriptorPool(
        vk::DescriptorPoolCreateInfo{.maxSets = poolSize.descriptorCount,
                                     .poolSizeCount = 1,
                                     .pPoolSizes = &poolSize});
}

} // namespace

App::App()
    : _window{WIDTH, HEIGHT, "prosper"}, _device{std::make_shared<Device>(
                                             _window.ptr())},
      _swapConfig{_device, {_window.width(), _window.height()}},
      _descriptorPool{createDescriptorPool(_device, _swapConfig.imageCount)},
      _cam{_device, _descriptorPool, _swapConfig.imageCount,
           vk::ShaderStageFlagBits::eVertex |
               vk::ShaderStageFlagBits::eFragment},
      _world{_device, _swapConfig.imageCount,
             resPath("glTF/FlightHelmet/glTF/FlightHelmet.gltf")},
      _swapchain{_device, _swapConfig}, _renderer{_device, _swapConfig,
                                                  _cam.descriptorSetLayout(),
                                                  _world._dsLayouts} {
    _cam.lookAt(vec3{0.25f, 0.2f, 0.75f}, vec3{0.f}, vec3{0.f, 1.f, 0.f});
    _cam.perspective(radians(CAMERA_FOV),
                     _window.width() / static_cast<float>(_window.height()),
                     CAMERA_NEAR, CAMERA_FAR);
}

App::~App() { _device->logical().destroy(_descriptorPool); }

void App::run() {
    while (_window.open()) {
        _window.startFrame();

        const auto &mouse = InputHandler::instance().mouse();
        if (mouse.leftDown && mouse.currentPos != mouse.lastPos) {
            _cam.orbit(mouse.currentPos, mouse.lastPos,
                       vec2(_window.width(), _window.height()) / 2.f);
        }
        if (mouse.rightDown && mouse.currentPos != mouse.lastPos) {
            _cam.scaleOrbit(mouse.currentPos.y, mouse.lastPos.y,
                            _window.height() / 2.f);
        }
        drawFrame();
    }

    // Wait for in flight rendering actions to finish
    _device->logical().waitIdle();
}

void App::recreateSwapchainAndRelated() {
    while (_window.width() == 0 && _window.height() == 0) {
        // Window is minimized so wait until its not
        glfwWaitEvents();
    }
    // Wait for resources to be out of use
    _device->logical().waitIdle();

    _swapConfig = SwapchainConfig{_device, {_window.width(), _window.height()}};

    _swapchain = Swapchain{_device, _swapConfig};
    _renderer.recreateSwapchainRelated(_swapConfig, _cam.descriptorSetLayout(),
                                       _world._dsLayouts);

    _cam.perspective(radians(CAMERA_FOV),
                     _window.width() / static_cast<float>(_window.height()),
                     CAMERA_NEAR, CAMERA_FAR);
}

void App::drawFrame() {
    // Corresponds to the logical swapchain frame [0, MAX_FRAMES_IN_FLIGHT)
    const size_t nextFrame = _swapchain.nextFrame();
    // Corresponds to the swapchain image
    const auto nextImage = [&] {
        const auto imageAvailable = _renderer.imageAvailable(nextFrame);
        auto nextImage = _swapchain.acquireNextImage(imageAvailable);
        while (!nextImage.has_value()) {
            // Recreate the swap chain as necessary
            recreateSwapchainAndRelated();
            nextImage = _swapchain.acquireNextImage(imageAvailable);
        }

        return nextImage.value();
    }();

    const auto signalSemaphores =
        _renderer.drawFrame(_world, _cam, _swapchain, nextImage);

    // Recreate swapchain if so indicated and explicitly handle resizes
    if (!_swapchain.present(signalSemaphores) || _window.resized())
        recreateSwapchainAndRelated();
}
