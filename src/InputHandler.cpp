#include "InputHandler.hpp"

InputHandler &InputHandler::instance() {
    static InputHandler ih;
    return ih;
}

const InputHandler::Mouse &InputHandler::mouse() const { return _mouse; }
