#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "App.hpp"

int main() {
    App app;

    try {
        app.init();
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
