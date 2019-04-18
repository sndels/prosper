#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "App.hpp"

int main() {
    App app;

#ifdef NDEBUG
    try {
        app.init();
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
#else
    app.init();
    app.run();
#endif // DEBUG

    return EXIT_SUCCESS;
}
