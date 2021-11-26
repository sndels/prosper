#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "App.hpp"

int main(int argc, char *argv[])
{
    std::filesystem::path scenePath{"glTF/FlightHelmet/glTF/FlightHelmet.gltf"};
    if (argc == 2)
    {
        scenePath = argv[1];
    }
    else if (argc != 1)
    {
        throw std::runtime_error(
            "Expected 0 or 1 cli args, got " + std::to_string(argc - 1));
    }

    App app{scenePath};
    app.run();

    return EXIT_SUCCESS;
}
