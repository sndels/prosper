#include "Window.hpp"

#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <iostream>

#include "InputHandler.hpp"

using namespace glm;

Window::Window(const uint32_t width, const uint32_t height,
               const std::string &title)
    : _width(width), _height(height) {
    glfwSetErrorCallback(Window::errorCallback);

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window =
        glfwCreateWindow(_width, _height, title.c_str(), nullptr, nullptr);

    glfwSetWindowUserPointer(_window, this);
    glfwSetKeyCallback(_window, Window::keyCallback);
    glfwSetCharCallback(_window, Window::charCallback);
    glfwSetCursorPosCallback(_window, Window::cursorPosCallback);
    glfwSetScrollCallback(_window, Window::scrollCallback);
    glfwSetMouseButtonCallback(_window, Window::mouseButtonCallback);
    glfwSetFramebufferSizeCallback(_window, Window::framebufferSizeCallback);
}

Window::~Window() {
    glfwDestroyWindow(_window);
    glfwTerminate();
}

GLFWwindow *Window::ptr() const { return _window; }

bool Window::open() const { return !glfwWindowShouldClose(_window); }

uint32_t Window::width() const { return _width; }

uint32_t Window::height() const { return _height; }

bool Window::resized() const { return _resized; }

void Window::startFrame() {
    _resized = false;
    auto &mouse = InputHandler::instance()._mouse;
    const vec2 lastMouse = mouse.currentPos;

    glfwPollEvents();

    // Stationary mouse is not handled by callbacks
    if (lastMouse == mouse.currentPos)
        mouse.lastPos = mouse.currentPos;
}

void Window::errorCallback(int error, const char *description) {
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

void Window::keyCallback(GLFWwindow *window, int32_t key, int32_t scancode,
                         int32_t action, int32_t mods) {
    (void)scancode;
    (void)mods;

    if (!ImGui::IsAnyItemActive()) {
        if (action == GLFW_PRESS) {
            switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            default:
                break;
            }
        }
    }
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void Window::charCallback(GLFWwindow *window, unsigned int c) {
    ImGui_ImplGlfw_CharCallback(window, c);
}

void Window::cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    (void)window;

    auto &mouse = InputHandler::instance()._mouse;
    mouse.lastPos = mouse.currentPos;
    mouse.currentPos = vec2(xpos, ypos);
}

void Window::scrollCallback(GLFWwindow *window, double xoffset,
                            double yoffset) {
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    else
        ;
}

void Window::mouseButtonCallback(GLFWwindow *window, int button, int action,
                                 int mods) {
    (void)window;
    (void)mods;

    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    } else {
        auto &mouse = InputHandler::instance()._mouse;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS && !io.WantCaptureMouse)
                mouse.leftDown = true;
            else
                mouse.leftDown = false;
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS && !io.WantCaptureMouse)
                mouse.rightDown = true;
            else
                mouse.rightDown = false;
        }
    }
}

void Window::framebufferSizeCallback(GLFWwindow *window, int width,
                                     int height) {
    Window *thisPtr = (Window *)glfwGetWindowUserPointer(window);
    auto uw = static_cast<uint32_t>(width);
    auto uh = static_cast<uint32_t>(height);
    if (thisPtr->_width != uw || thisPtr->_height != uh) {
        thisPtr->_width = uw;
        thisPtr->_height = uh;
        thisPtr->_resized = true;
    }
}
