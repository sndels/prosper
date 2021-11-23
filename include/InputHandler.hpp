#ifndef PROSPER_INPUTHANDLER_HPP
#define PROSPER_INPUTHANDLER_HPP

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

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
