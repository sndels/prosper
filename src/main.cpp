#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <wheels/allocators/linear_allocator.hpp>

#include "App.hpp"
#include "Utils.hpp"

using namespace wheels;

static const char *const default_scene_path =
    "glTF/FlightHelmet/glTF/FlightHelmet.gltf";

int main(int argc, char *argv[])
{
    try
    {
        std::filesystem::path scenePath{default_scene_path};
        bool enableDebugLayers = false;
        if (argc == 2)
        {
            if (StrSpan{argv[1]} == "--debugLayers")
                enableDebugLayers = true;
            else
                scenePath = argv[1];
        }
        else if (argc == 3)
        {
            if (StrSpan{argv[1]} == "--debugLayers")
                enableDebugLayers = true;
            else
                throw std::runtime_error(
                    "Unexpected argument '" + std::string{argv[1]} + "'");
            scenePath = argv[2];
        }
        else if (argc != 1)
        {
            throw std::runtime_error(
                "Expected 0-2 cli args, got " + std::to_string(argc - 1));
        }

        if (scenePath == default_scene_path)
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

        LinearAllocator scratchBacking{megabytes(256)};

        App app{ScopedScratch{scratchBacking}, scenePath, enableDebugLayers};
        app.run();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception thrown: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}
