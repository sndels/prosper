#ifndef PROSPER_INPUTHANDLER_HPP
#define PROSPER_INPUTHANDLER_HPP

#include <glm/glm.hpp>

class InputHandler
{
  public:
    struct Mouse
    {
        glm::vec2 currentPos;
        glm::vec2 lastPos;
        bool leftDown;
        bool rightDown;

        Mouse()
        : currentPos(0.f)
        , lastPos(0.f)
        , leftDown(false)
        , rightDown(false)
        {
        }
    };

    InputHandler(const InputHandler &other) = delete;
    InputHandler &operator=(const InputHandler &other) = delete;

    static InputHandler &instance();
    const Mouse &mouse() const;

  private:
    InputHandler() = default;

    Mouse _mouse;

    // Window updates input state directly
    friend class Window;
};

#endif // PROSPER_INPUTHANDLER_HPP
