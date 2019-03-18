#include "Window.hpp"

#include <iostream>

Window::~Window()
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void Window::init(const uint32_t width, const uint32_t height, const std::string& title)
{
    _width = width;
    _height = height;

    glfwSetErrorCallback(Window::errorCallback);

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window = glfwCreateWindow(_width, _height, title.c_str(), nullptr, nullptr);

    glfwSetWindowUserPointer(_window, this);
    glfwSetKeyCallback(_window, Window::keyCallback);
    glfwSetFramebufferSizeCallback(_window, Window::framebufferSizeCallback);
}

GLFWwindow* Window::ptr() const
{
    return _window;
}

bool Window::open() const
{
    return !glfwWindowShouldClose(_window);
}

uint32_t Window::width() const
{
    return _width;
}

uint32_t Window::height() const
{

    return _height;
}

bool Window::resized() const
{
    return _resized;
}

void Window::startFrame()
{
    _resized = false;
    glfwPollEvents();
}

void Window::errorCallback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

void Window::keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
    (void) scancode;
    (void) mods;

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

void Window::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    Window* thisPtr = (Window*)glfwGetWindowUserPointer(window);
    auto uw = static_cast<uint32_t>(width);
    auto uh = static_cast<uint32_t>(height);
    if (thisPtr->_width != uw || thisPtr->_height != uh) {
        thisPtr->_width = uw;
        thisPtr->_height = uh;
        thisPtr->_resized = true;
    }
}
