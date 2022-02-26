#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "App.hpp"
#include "Utils.hpp"

int main(int argc, char *argv[])
{
    try
    {
        std::filesystem::path scenePath{
            "glTF/FlightHelmet/glTF/FlightHelmet.gltf"};
        if (argc == 2)
        {
            scenePath = argv[1];
        }
        else if (argc != 1)
        {
            throw std::runtime_error(
                "Expected 0 or 1 cli args, got " + std::to_string(argc - 1));
        }
        else
        {
            const auto sceneConfPath = resPath("scene.txt");
            if (std::filesystem::exists(sceneConfPath))
            {
                std::ifstream file{sceneConfPath};
                std::string scenePathStr;
                std::getline(file, scenePathStr);
                scenePath = scenePathStr;
            }
        }

        App app{scenePath};
        app.run();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception thrown: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}
