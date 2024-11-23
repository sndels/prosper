#include "Window.hpp"

#include "Allocators.hpp"
#include "utils/InputHandler.hpp"
#include "utils/Logger.hpp"
#include "utils/Utils.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

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
    alloc->deallocate(block);
}

} // namespace

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Window gWindow;

Window::~Window()
{
    WHEELS_ASSERT(
        (!m_initialized || m_window == nullptr) && "destroy() not called");
}

void Window::init(
    const Pair<uint32_t, uint32_t> &resolution, const char *title) noexcept
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating window");

    m_width = resolution.first;
    m_height = resolution.second;

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

    m_window = glfwCreateWindow(
        asserted_cast<int>(m_width), asserted_cast<int>(m_height), title,
        nullptr, nullptr);

    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, Window::keyCallback);
    glfwSetCharCallback(m_window, Window::charCallback);
    glfwSetCursorPosCallback(m_window, Window::cursorPosCallback);
    glfwSetCursorEnterCallback(m_window, Window::cursorEnterCallback);
    glfwSetScrollCallback(m_window, Window::scrollCallback);
    glfwSetMouseButtonCallback(m_window, Window::mouseButtonCallback);
    glfwSetFramebufferSizeCallback(m_window, Window::framebufferSizeCallback);

    // Non-raw input virtual mouse position seems to jump aroud much more on
    // Win10. First callback after disabling cursor could be 100s of px away
    // from the click position if drag is initiated during a fast move.
    if (glfwRawMouseMotionSupported() == GLFW_TRUE)
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

#ifdef _WIN32
    // Try to set dark mode to match the inactive title bar color to the ui
    // color scheme
    // https://stackoverflow.com/a/70693198
    HWND hwnd = glfwGetWin32Window(m_window);
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

    m_initialized = true;
}

void Window::destroy() noexcept
{
    glfwDestroyWindow(m_window);
    glfwTerminate();

    // m_initialized = true and m_window = nullptr mark a destroyed window
    m_window = nullptr;
}

GLFWwindow *Window::ptr() const { return m_window; }

bool Window::open() const
{
    return glfwWindowShouldClose(m_window) == GLFW_FALSE;
}

uint32_t Window::width() const { return m_width; }

uint32_t Window::height() const { return m_height; }

bool Window::resized() const { return m_resized; }

void Window::startFrame()
{
    m_resized = false;

    glfwPollEvents();
    gInputHandler.handleKeyStateUpdate();
}

void Window::pollCursorPosition() const
{
    const CursorState cursor = gInputHandler.cursor();
    double x = static_cast<double>(cursor.position.x);
    double y = static_cast<double>(cursor.position.y);
    glfwGetCursorPos(m_window, &x, &y);
    gInputHandler.handleMouseMove(x, y);
}

void Window::errorCallback(int error, const char *description)
{
    LOG_ERR("GLFW error %d: %s", error, description);
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
    if (thisPtr->m_width != uw || thisPtr->m_height != uh)
    {
        thisPtr->m_width = uw;
        thisPtr->m_height = uh;
        thisPtr->m_resized = true;
    }
}
