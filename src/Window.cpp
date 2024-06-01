#include "Window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <iostream>

#include "Allocators.hpp"
#include "utils/InputHandler.hpp"
#include "utils/Utils.hpp"

#ifdef _WIN32

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>

#endif // _WIN32

using namespace wheels;

namespace
{

void *allocatefun(size_t size, void *user)
{
    WHEELS_ASSERT(user != nullptr);
    TlsfAllocator *alloc = static_cast<TlsfAllocator *>(user);
    return alloc->allocate(size);
}

void *reallocatefun(void *block, size_t size, void *user)
{
    WHEELS_ASSERT(user != nullptr);
    TlsfAllocator *alloc = static_cast<TlsfAllocator *>(user);
    return alloc->reallocate(block, size);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) glfw interface
void deallocatefun(void *block, void *user)
{
    WHEELS_ASSERT(user != nullptr);
    TlsfAllocator *alloc = static_cast<TlsfAllocator *>(user);
    return alloc->deallocate(block);
}

} // namespace

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Window gWindow;

Window::~Window()
{
    WHEELS_ASSERT(
        (!_initialized || _window == nullptr) && "destroy() not called");
}

void Window::init(
    const Pair<uint32_t, uint32_t> &resolution, const char *title) noexcept
{
    WHEELS_ASSERT(!_initialized);

    printf("Creating window\n");

    _width = resolution.first;
    _height = resolution.second;

    GLFWallocator allocator;
    allocator.allocate = allocatefun;
    allocator.reallocate = reallocatefun;
    allocator.deallocate = deallocatefun;
    static_assert(std::is_same_v<decltype(gAllocators.general), TlsfAllocator>);
    allocator.user = &gAllocators.general;

    glfwInitAllocator(&allocator);

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

    _initialized = true;
}

void Window::destroy() noexcept
{
    glfwDestroyWindow(_window);
    glfwTerminate();

    // _initialized = true and _window = nullptr mark a destroyed window
    _window = nullptr;
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
    gInputHandler.handleKeyStateUpdate();
}

void Window::pollCursorPosition() const
{
    const CursorState cursor = gInputHandler.cursor();
    double x = static_cast<double>(cursor.position.x);
    double y = static_cast<double>(cursor.position.y);
    glfwGetCursorPos(_window, &x, &y);
    gInputHandler.handleMouseMove(x, y);
}

void Window::errorCallback(int error, const char *description)
{
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

void Window::keyCallback(
    GLFWwindow *window, int32_t key, int32_t scancode, int32_t action,
    int32_t mods)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    if (!ImGui::IsAnyItemActive())
        gInputHandler.handleKey(key, scancode, action, mods);
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void Window::charCallback(GLFWwindow *window, unsigned int c)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    ImGui_ImplGlfw_CharCallback(window, c);
}

void Window::cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    gInputHandler.handleMouseMove(xpos, ypos);
}

void Window::cursorEnterCallback(GLFWwindow *window, int entered)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    gInputHandler.handleCursorEntered(entered == GL_TRUE);
}

void Window::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    else
        gInputHandler.handleMouseScroll(xoffset, yoffset);
}

void Window::mouseButtonCallback(
    GLFWwindow *window, int button, int action, int mods)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    // Make sure we don't just drop camera drag end events when mouse moves over
    // a UI element
    if (!io.WantCaptureMouse || action == GLFW_RELEASE)
        gInputHandler.handleMouseButton(button, action, mods);
}

// NOLINTNEXTLINE mirrors the glfw interface
void Window::framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    auto *thisPtr = static_cast<Window *>(glfwGetWindowUserPointer(window));
    WHEELS_ASSERT(window == thisPtr->ptr());

    auto uw = asserted_cast<uint32_t>(width);
    auto uh = asserted_cast<uint32_t>(height);
    if (thisPtr->_width != uw || thisPtr->_height != uh)
    {
        thisPtr->_width = uw;
        thisPtr->_height = uh;
        thisPtr->_resized = true;
    }
}
