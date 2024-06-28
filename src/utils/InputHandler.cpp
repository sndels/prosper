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
    if (m_mouseGesture.has_value() &&
        m_mouseGesture->type == MouseGestureType::TrackZoom)
    {
        m_mouseGesture.reset();
    }
}

const CursorState &InputHandler::cursor() const { return m_cursor; }

const StaticArray<KeyState, KeyCount> &InputHandler::keyboard() const
{
    return m_keyboard;
}

const wheels::Optional<MouseGesture> &InputHandler::mouseGesture() const
{
    return m_mouseGesture;
}

void InputHandler::handleCursorEntered(bool entered)
{
    m_cursor.inside = entered;
}

void InputHandler::handleMouseScroll(double xoffset, double yoffset)
{
    (void)xoffset;
    if (!m_mouseGesture.has_value())
    {
        m_mouseGesture = MouseGesture{
            .verticalScroll = static_cast<float>(yoffset),
            .type = MouseGestureType::TrackZoom,
        };
    }
    else if (m_mouseGesture->type == MouseGestureType::TrackZoom)
    {
        m_mouseGesture->verticalScroll += static_cast<float>(yoffset);
    }
}

// NOLINTNEXTLINE mirrors the glfw interface
void InputHandler::handleMouseButton(int button, int action, int /*mods*/)
{
    if (m_cursor.inside)
    {
        if (m_mouseGesture.has_value())
        {
            if (action == GLFW_RELEASE)
            {
                m_mouseGesture.reset();
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
                if (m_keyboard[KeyAlt] == KeyState::Held)
                {
                    m_mouseGesture = MouseGesture{
                        .startPos = m_cursor.position,
                        .currentPos = m_cursor.position,
                        .type = MouseGestureType::TrackPlane,
                    };
                }
                else
                {
                    m_mouseGesture = MouseGesture{
                        .startPos = m_cursor.position,
                        .currentPos = m_cursor.position,
                        .type = MouseGestureType::TrackBall,
                    };
                }
                // Constrain mouse so that drags aren't bounded by the window
                // size
                GLFWwindow *window = gWindow.ptr();
                WHEELS_ASSERT(window != nullptr);
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                m_cursor.shown = false;
            }
            else if (
                button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS &&
                m_keyboard[KeyCtrl] == KeyState::Held)
            {
                m_mouseGesture = MouseGesture{
                    .startPos = m_cursor.position,
                    .currentPos = m_cursor.position,
                    .type = MouseGestureType::SelectPoint,
                };
            }
        }
    }
}

void InputHandler::handleMouseMove(double xpos, double ypos)
{
    m_cursor.position = vec2(xpos, ypos);
    if (m_mouseGesture.has_value())
    {
        m_mouseGesture->currentPos = m_cursor.position;
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
        m_keyboard[key] = KeyState::Pressed;
        m_keyboardUpdated[key] = true;
    }
    else if (action == GLFW_RELEASE)
    {
        m_keyboard[key] = KeyState::Released;
        m_keyboardUpdated[key] = true;
    }
}

void InputHandler::handleKeyStateUpdate()
{
    for (uint8_t key = 0; key < KeyCount; ++key)
    {
        if (!m_keyboardUpdated[key])
        {
            KeyState &state = m_keyboard[key];
            if (state == KeyState::Pressed)
                state = KeyState::Held;
            else if (state == KeyState::Released)
                state = KeyState::Neutral;
        }
    }
    m_keyboardUpdated = StaticArray<bool, KeyCount>{false};
}

void InputHandler::hideCursor()
{
    if (m_cursor.shown)
    {
        GLFWwindow *window = gWindow.ptr();
        WHEELS_ASSERT(window != nullptr);
        // No need to check for active gesture as cursor is always hidden during
        // gestures
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        m_cursor.shown = false;
    }
}

void InputHandler::showCursor()
{
    if (!m_cursor.shown)
    {
        // Gestures' disabled cursor have precedence
        if (!m_mouseGesture.has_value())
        {
            GLFWwindow *window = gWindow.ptr();
            WHEELS_ASSERT(window != nullptr);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            m_cursor.shown = true;
        }
    }
}
