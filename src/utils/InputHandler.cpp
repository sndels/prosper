#include "InputHandler.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

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
    default:
        return KeyNotMapped;
    }
}

} // namespace

InputHandler &InputHandler::instance()
{
    static InputHandler ih;
    return ih;
}

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
void InputHandler::handleMouseButton(int button, int action, int mods)
{
    if (_cursor.inside)
    {
        if (_mouseGesture.has_value())
        {
            if (action == GLFW_RELEASE)
                _mouseGesture.reset();
        }
        else
        {
            if ((button == GLFW_MOUSE_BUTTON_MIDDLE ||
                 button == GLFW_MOUSE_BUTTON_RIGHT) &&
                action == GLFW_PRESS)
            {
                if ((mods & GLFW_MOD_SHIFT) == GLFW_MOD_SHIFT)
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
    _keyboardUpdated = {false};
}
