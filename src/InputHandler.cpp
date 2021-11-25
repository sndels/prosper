#include "InputHandler.hpp"

#include <GLFW/glfw3.h>

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/glm.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

using namespace glm;

InputHandler &InputHandler::instance()
{
    static InputHandler ih;
    return ih;
}

void InputHandler::clearSingleFrameGestures()
{
    if (_mouseGesture && _mouseGesture->type == MouseGestureType::TrackZoom)
    {
        _mouseGesture = std::nullopt;
    }
}

const CursorState &InputHandler::cursor() const { return _cursor; }

const ModifierState &InputHandler::modifiers() const { return _modifiers; }

const std::optional<MouseGesture> &InputHandler::mouseGesture() const
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
    if (!_mouseGesture)
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

void InputHandler::handleMouseButton(int button, int action, int mods)
{
    if (_cursor.inside)
    {
        if (_mouseGesture)
        {
            if (action == GLFW_RELEASE)
                _mouseGesture = std::nullopt;
        }
        else
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
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
    if (_mouseGesture)
    {
        _mouseGesture->currentPos = _cursor.position;
    }
}
