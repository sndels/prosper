#ifndef PROSPER_WINDOW_HPP
#define PROSPER_WINDOW_HPP

#include <GLFW/glfw3.h>

#include <string>

class Window
{
  public:
    Window(
        const std::pair<uint32_t, uint32_t> &resolution,
        const std::string &title);
    ~Window();

    Window(const Window &other) = delete;
    Window(Window &&other) = delete;
    Window &operator=(const Window &other) = delete;
    Window &operator=(Window &&other) = delete;

    [[nodiscard]] GLFWwindow *ptr() const;
    [[nodiscard]] bool open() const;
    [[nodiscard]] uint32_t width() const;
    [[nodiscard]] uint32_t height() const;
    [[nodiscard]] bool resized() const;

    void startFrame();

    static void errorCallback(int error, const char *description);
    static void keyCallback(
        GLFWwindow *window, int32_t key, int32_t scancode, int32_t action,
        int32_t mods);
    static void charCallback(GLFWwindow *window, unsigned int c);
    static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos);
    static void cursorEnterCallback(GLFWwindow *window, int entered);
    static void scrollCallback(
        GLFWwindow *window, double xoffset, double yoffset);
    static void mouseButtonCallback(
        GLFWwindow *window, int button, int action, int mods);
    static void framebufferSizeCallback(
        GLFWwindow *window, int width, int height);

  private:
    GLFWwindow *_window = nullptr;
    uint32_t _width = 0;
    uint32_t _height = 0;
    bool _resized = false;
};

#endif // PROSPER_WINDOW_HPP
