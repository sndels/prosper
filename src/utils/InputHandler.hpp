#ifndef PROSPER_UTILS_INPUT_HANDLER_HPP
#define PROSPER_UTILS_INPUT_HANDLER_HPP

#include <glm/glm.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

enum class KeyState : uint8_t
{
    Neutral,  // Key is just hangin'
    Pressed,  // Key was pressed between previous and current frame
    Held,     // Key is being held down
    Released, // Key was released between previous and current frame
};

enum Key : uint8_t
{
    KeyI,
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    KeyQ,
    KeyE,
    KeyShift,
    KeyCtrl,
    KeyAlt,
    KeyCount,
    KeyNotMapped,
};

struct CursorState
{
    glm::vec2 position{0.f, 0.f};
    bool inside{false};
    bool shown{true};
};

enum class MouseGestureType : uint8_t
{
    TrackBall,
    TrackPlane,
    TrackZoom,
    SelectPoint,
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
    InputHandler() noexcept = default;
    ~InputHandler() = default;

    InputHandler(const InputHandler &other) = delete;
    InputHandler(InputHandler &&other) = delete;
    InputHandler &operator=(const InputHandler &other) = delete;
    InputHandler &operator=(InputHandler &&other) = delete;

    void clearSingleFrameGestures();

    [[nodiscard]] const CursorState &cursor() const;
    [[nodiscard]] const wheels::StaticArray<KeyState, KeyCount> &keyboard()
        const;
    [[nodiscard]] const wheels::Optional<MouseGesture> &mouseGesture() const;

    void handleCursorEntered(bool entered);
    void handleMouseScroll(double xoffset, double yoffset);
    void handleMouseButton(int button, int action, int mods);
    void handleMouseMove(double xpos, double ypos);
    void handleKey(int glfwKey, int scancode, int action, int mods);
    void handleKeyStateUpdate();

    void hideCursor();
    // This doesn't override hidden cursor when gesture is active
    void showCursor();

  private:
    CursorState m_cursor;
    wheels::StaticArray<KeyState, KeyCount> m_keyboard{KeyState::Neutral};
    wheels::StaticArray<bool, KeyCount> m_keyboardUpdated{false};
    wheels::Optional<MouseGesture> m_mouseGesture;
};

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern InputHandler gInputHandler;

#endif // PROSPER_UTILS_INPUT_HANDLER_HPP
