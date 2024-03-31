#include "Window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <iostream>

#include "utils/InputHandler.hpp"
#include "utils/Utils.hpp"

#ifdef _WIN32

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>

#endif // _WIN32

using namespace wheels;

Window::Window(
    const Pair<uint32_t, uint32_t> &resolution, const char *title,
    InputHandler *inputHandler) noexcept
: _inputHandler{inputHandler}
, _width{resolution.first}
, _height{resolution.second}
{
    WHEELS_ASSERT(_inputHandler != nullptr);

    printf("Creating window\n");

    glfwSetErrorCallback(Window::errorCallback);

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window = glfwCreateWindow(
        asserted_cast<int>(_width), asserted_cast<int>(_height), title, nullptr,
        nullptr);

    glfwSetWindowUserPointer(_window, this);
    glfwSetKeyCallback(_window, Window::keyCallback);
    glfwSetCharCallback(_window, Window::charCallback);
    glfwSetCursorPosCallback(_window, Window::cursorPosCallback);
    glfwSetCursorEnterCallback(_window, Window::cursorEnterCallback);
    glfwSetScrollCallback(_window, Window::scrollCallback);
    glfwSetMouseButtonCallback(_window, Window::mouseButtonCallback);
    glfwSetFramebufferSizeCallback(_window, Window::framebufferSizeCallback);

    // Non-raw input virtual mouse position seems to jump aroud much more on
    // Win10. First callback after disabling cursor could be 100s of px away
    // from the click position if drag is initiated during a fast move.
    if (glfwRawMouseMotionSupported() == GLFW_TRUE)
        glfwSetInputMode(_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

#ifdef _WIN32
    // Try to set dark mode to match the inactive title bar color to the ui
    // color scheme
    // https://stackoverflow.com/a/70693198
    HWND hwnd = glfwGetWin32Window(_window);
    const BOOL use_dark_mode = TRUE;
    // These aren't exposed in older SDKs but might still work
    const WORD DWMWA_USE_IMMERSIVE_DARK_MODE = 20;
    const WORD DWMWA_USE_IMMERSIVE_DARK_MODE_PRE_20H1 = 19;
    // It's ok if these fail so let's not bother checking for success
    DwmSetWindowAttribute(
        hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &use_dark_mode,
        sizeof(use_dark_mode));
    DwmSetWindowAttribute(
        hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_PRE_20H1, &use_dark_mode,
        sizeof(use_dark_mode));
#endif // _WIN32
}

Window::~Window()
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}

GLFWwindow *Window::ptr() const { return _window; }

bool Window::open() const
{
    return glfwWindowShouldClose(_window) == GLFW_FALSE;
}

uint32_t Window::width() const { return _width; }

uint32_t Window::height() const { return _height; }

bool Window::resized() const { return _resized; }

void Window::startFrame()
{
    _resized = false;

    glfwPollEvents();
    _inputHandler->handleKeyStateUpdate();
}

void Window::pollCursorPosition() const
{
    const CursorState cursor = _inputHandler->cursor();
    double x = static_cast<double>(cursor.position.x);
    double y = static_cast<double>(cursor.position.y);
    glfwGetCursorPos(_window, &x, &y);
    _inputHandler->handleMouseMove(x, y);
}

void Window::errorCallback(int error, const char *description)
{
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

void Window::keyCallback(
    GLFWwindow *window, int32_t key, int32_t scancode, int32_t action,
    int32_t mods)
{
    (void)scancode;
    (void)mods;

    if (!ImGui::IsAnyItemActive())
    {
        auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
        thisPtr->_inputHandler->handleKey(key, scancode, action, mods);
    }
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void Window::charCallback(GLFWwindow *window, unsigned int c)
{
    ImGui_ImplGlfw_CharCallback(window, c);
}

void Window::cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    thisPtr->_inputHandler->handleMouseMove(xpos, ypos);
}

void Window::cursorEnterCallback(GLFWwindow *window, int entered)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    thisPtr->_inputHandler->handleCursorEntered(entered == GL_TRUE);
}

void Window::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    else
    {
        auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
        thisPtr->_inputHandler->handleMouseScroll(xoffset, yoffset);
    }
}

void Window::mouseButtonCallback(
    GLFWwindow *window, int button, int action, int mods)
{
    (void)mods;

    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    // Make sure we don't just drop camera drag end events when mouse moves over
    // a UI element
    if (!io.WantCaptureMouse || action == GLFW_RELEASE)
    {
        auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
        thisPtr->_inputHandler->handleMouseButton(window, button, action, mods);
    }
}

// NOLINTNEXTLINE mirrors the glfw interface
void Window::framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    auto uw = asserted_cast<uint32_t>(width);
    auto uh = asserted_cast<uint32_t>(height);
    if (thisPtr->_width != uw || thisPtr->_height != uh)
    {
        thisPtr->_width = uw;
        thisPtr->_height = uh;
        thisPtr->_resized = true;
    }
}
