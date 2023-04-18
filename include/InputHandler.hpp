#ifndef PROSPER_INPUTHANDLER_HPP
#define PROSPER_INPUTHANDLER_HPP

#include <glm/glm.hpp>

#include <wheels/containers/optional.hpp>

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
    InputHandler(InputHandler &&other) = delete;
    InputHandler &operator=(const InputHandler &other) = delete;
    InputHandler &operator=(InputHandler &&other) = delete;

    static InputHandler &instance();

    void clearSingleFrameGestures();

    [[nodiscard]] const CursorState &cursor() const;
    [[nodiscard]] const ModifierState &modifiers() const;
    [[nodiscard]] const wheels::Optional<MouseGesture> &mouseGesture() const;

    void handleCursorEntered(bool entered);
    void handleMouseScroll(double xoffset, double yoffset);
    void handleMouseButton(int button, int action, int mods);
    void handleMouseMove(double xpos, double ypos);

  private:
    InputHandler() = default;
    ~InputHandler() = default;

    CursorState _cursor;
    ModifierState _modifiers;
    wheels::Optional<MouseGesture> _mouseGesture;
};

#endif // PROSPER_INPUTHANDLER_HPP
