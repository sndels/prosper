#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "App.hpp"

int main() {
    App app;
    app.init();
    app.run();

    return EXIT_SUCCESS;
}
