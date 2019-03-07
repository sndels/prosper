#ifndef PROSPER_WINDOW_HPP
#define PROSPER_WINDOW_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>

class Window {
public:
    Window() = default;
    ~Window();

    Window(const Window& other) = delete;
    Window operator=(const Window& other) = delete;

    void init(uint32_t width, uint32_t height, const std::string& title);

    GLFWwindow* ptr();
    bool open() const;
    uint32_t width() const;
    uint32_t height() const;
    bool resized() const;

    void startFrame();

    static void errorCallback(int error, const char* description);
    static void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

private:
    GLFWwindow* _window = nullptr;
    uint32_t _width = 0;
    uint32_t _height = 0;
    bool _resized = false;

};

#endif // PROSPER_WINDOW_HPP
