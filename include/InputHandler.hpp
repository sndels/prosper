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

#include <optional>

struct ModifierState
{
    bool shift{false};
};

struct CursorState
{
    glm::vec2 position{0.f, 0.f};
    bool inside{false};
};

enum class MouseGestureType
{
    TrackBall,
    TrackPlane,
    TrackZoom,
};

struct MouseGesture
{
    glm::vec2 startPos{0.f, 0.f};
    glm::vec2 currentPos{0.f, 0.f};
    float verticalScroll{0.f};
    MouseGestureType type;
};

class InputHandler
{
  public:
    InputHandler(const InputHandler &other) = delete;
    InputHandler &operator=(const InputHandler &other) = delete;

    static InputHandler &instance();

    void clearSingleFrameGestures();

    const CursorState &cursor() const;
    const ModifierState &modifiers() const;
    const std::optional<MouseGesture> &mouseGesture() const;

    void handleCursorEntered(bool entered);
    void handleMouseScroll(double xoffset, double yoffset);
    void handleMouseButton(int button, int action, int mods);
    void handleMouseMove(double xpos, double ypos);

  private:
    InputHandler() = default;

    CursorState _cursor;
    ModifierState _modifiers;
    std::optional<MouseGesture> _mouseGesture;
};

#endif // PROSPER_INPUTHANDLER_HPP
