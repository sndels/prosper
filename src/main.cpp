#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cxxopts.hpp>
#include <wheels/allocators/linear_allocator.hpp>

#include "App.hpp"
#include "Utils.hpp"

using namespace wheels;

namespace
{
const char *const s_default_scene_path =
    "glTF/FlightHelmet/glTF/FlightHelmet.gltf";

// NOLINTNEXTLINE(*-avoid-c-arrays): Mandatory
App::Settings parseCli(int argc, char *argv[])
{
    cxxopts::Options options("prosper", "A toy Vulkan renderer");
    // clang-format off
        options.add_options()
            ("debugLayers", "Enable Vulkan debug layers")
            ("dumpShaderDisassembly", "Dump shader disassembly to stdout")
            ("disableDeferredLoading", "Load all assets up front")
            ("sceneFile", std::string{"Scene to open (default: '"} + s_default_scene_path +"')",
             cxxopts::value<std::string>()->default_value(""));
    // clang-format on
    options.parse_positional({"sceneFile"});
    const cxxopts::ParseResult args = options.parse(argc, argv);

    std::filesystem::path scenePath{args["sceneFile"].as<std::string>()};
    if (scenePath.empty())
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
    if (scenePath.empty())
        scenePath = s_default_scene_path;

    return App::Settings{
        .scene = scenePath,
        .deferredLoading = !args["disableDeferredLoading"].as<bool>(),
        .device =
            Device::Settings{
                .enableDebugLayers = args["debugLayers"].as<bool>(),
                .dumpShaderDisassembly =
                    args["dumpShaderDisassembly"].as<bool>(),
            },
    };
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const App::Settings settings = parseCli(argc, argv);

        App app{settings};
        app.run();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception thrown: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}
