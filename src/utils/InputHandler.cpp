#include "InputHandler.hpp"

#include "../Window.hpp"

#include <GLFW/glfw3.h>

using namespace glm;
using namespace wheels;

namespace
{

Key convertKey(int glfwCode)
{
    switch (glfwCode)
    {
    case GLFW_KEY_I:
        return KeyI;
    case GLFW_KEY_W:
        return KeyW;
    case GLFW_KEY_A:
        return KeyA;
    case GLFW_KEY_S:
        return KeyS;
    case GLFW_KEY_D:
        return KeyD;
    case GLFW_KEY_Q:
        return KeyQ;
    case GLFW_KEY_E:
        return KeyE;
    case GLFW_KEY_LEFT_SHIFT:
    case GLFW_KEY_RIGHT_SHIFT:
        return KeyShift;
    case GLFW_KEY_LEFT_CONTROL:
    case GLFW_KEY_RIGHT_CONTROL:
        return KeyCtrl;
    case GLFW_KEY_LEFT_ALT:
    case GLFW_KEY_RIGHT_ALT:
        return KeyAlt;
    default:
        return KeyNotMapped;
    }
}

} // namespace

// This depended on by Window and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
InputHandler gInputHandler;

void InputHandler::clearSingleFrameGestures()
{
    if (_mouseGesture.has_value() &&
        _mouseGesture->type == MouseGestureType::TrackZoom)
    {
        _mouseGesture.reset();
    }
}

const CursorState &InputHandler::cursor() const { return _cursor; }

const StaticArray<KeyState, KeyCount> &InputHandler::keyboard() const
{
    return _keyboard;
}

const wheels::Optional<MouseGesture> &InputHandler::mouseGesture() const
{
    return _mouseGesture;
}

void InputHandler::handleCursorEntered(bool entered)
{
    _cursor.inside = entered;
}

void InputHandler::handleMouseScroll(double xoffset, double yoffset)
{
    (void)xoffset;
    if (!_mouseGesture.has_value())
    {
        _mouseGesture = MouseGesture{
            .verticalScroll = static_cast<float>(yoffset),
            .type = MouseGestureType::TrackZoom,
        };
    }
    else if (_mouseGesture->type == MouseGestureType::TrackZoom)
    {
        _mouseGesture->verticalScroll += static_cast<float>(yoffset);
    }
}

// NOLINTNEXTLINE mirrors the glfw interface
void InputHandler::handleMouseButton(int button, int action, int /*mods*/)
{
    if (_cursor.inside)
    {
        if (_mouseGesture.has_value())
        {
            if (action == GLFW_RELEASE)
            {
                _mouseGesture.reset();
                // Restore normal mouse input
                showCursor();
            }
        }
        else
        {
            if ((button == GLFW_MOUSE_BUTTON_MIDDLE ||
                 button == GLFW_MOUSE_BUTTON_RIGHT) &&
                action == GLFW_PRESS)
            {
                if (_keyboard[KeyAlt] == KeyState::Held)
                {
                    _mouseGesture = MouseGesture{
                        .startPos = _cursor.position,
                        .currentPos = _cursor.position,
                        .type = MouseGestureType::TrackPlane,
                    };
                }
                else
                {
                    _mouseGesture = MouseGesture{
                        .startPos = _cursor.position,
                        .currentPos = _cursor.position,
                        .type = MouseGestureType::TrackBall,
                    };
                }
                // Constrain mouse so that drags aren't bounded by the window
                // size
                GLFWwindow *window = gWindow.ptr();
                WHEELS_ASSERT(window != nullptr);
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                _cursor.shown = false;
            }
            else if (
                button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS &&
                _keyboard[KeyCtrl] == KeyState::Held)
            {
                _mouseGesture = MouseGesture{
                    .startPos = _cursor.position,
                    .currentPos = _cursor.position,
                    .type = MouseGestureType::SelectPoint,
                };
            }
        }
    }
}

void InputHandler::handleMouseMove(double xpos, double ypos)
{
    _cursor.position = vec2(xpos, ypos);
    if (_mouseGesture.has_value())
    {
        _mouseGesture->currentPos = _cursor.position;
    }
}

// NOLINTNEXTLINE mirrors the glfw interface
void InputHandler::handleKey(
    int glfwKey, int /*scancode*/, int action, int /*mods*/)
{
    const Key key = convertKey(glfwKey);
    if (key == KeyNotMapped)
        return;

    if (action == GLFW_PRESS)
    {
        _keyboard[key] = KeyState::Pressed;
        _keyboardUpdated[key] = true;
    }
    else if (action == GLFW_RELEASE)
    {
        _keyboard[key] = KeyState::Released;
        _keyboardUpdated[key] = true;
    }
}

void InputHandler::handleKeyStateUpdate()
{
    for (uint8_t key = 0; key < KeyCount; ++key)
    {
        if (!_keyboardUpdated[key])
        {
            KeyState &state = _keyboard[key];
            if (state == KeyState::Pressed)
                state = KeyState::Held;
            else if (state == KeyState::Released)
                state = KeyState::Neutral;
        }
    }
    _keyboardUpdated = StaticArray<bool, KeyCount>{false};
}

void InputHandler::hideCursor()
{
    if (_cursor.shown)
    {
        GLFWwindow *window = gWindow.ptr();
        WHEELS_ASSERT(window != nullptr);
        // No need to check for active gesture as cursor is always hidden during
        // gestures
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        _cursor.shown = false;
    }
}

void InputHandler::showCursor()
{
    if (!_cursor.shown)
    {
        // Gestures' disabled cursor have precedence
        if (!_mouseGesture.has_value())
        {
            GLFWwindow *window = gWindow.ptr();
            WHEELS_ASSERT(window != nullptr);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            _cursor.shown = true;
        }
    }
}
