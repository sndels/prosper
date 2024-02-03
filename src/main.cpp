#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cxxopts.hpp>
#include <wheels/allocators/linear_allocator.hpp>

#include "App.hpp"
#include "gfx/Device.hpp"
#include "utils/Utils.hpp"

#ifdef LIVEPP_PATH
#include "API/x64/LPP_API_x64_CPP.h"
#endif // LIVEPP_PATH

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
            ("breakOnValidationError", "Break debugger on Vulkan validation error")
            ("robustAccess", "Enable VK_EXT_robustness2 for buffers and images")
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
        .device =
            Device::Settings{
                .enableDebugLayers = args["debugLayers"].as<bool>(),
                .dumpShaderDisassembly =
                    args["dumpShaderDisassembly"].as<bool>(),
                .breakOnValidationError =
                    args["breakOnValidationError"].as<bool>(),
                .robustAccess = args["robustAccess"].as<bool>(),
            },
    };
}

} // namespace

int main(int argc, char *argv[])
{
#ifdef LIVEPP_PATH
    // create a default agent, loading the Live++ agent from the given path
    lpp::LppDefaultAgent lppAgent =
        lpp::LppCreateDefaultAgent(nullptr, LIVEPP_PATH);

    // bail out in case the agent is not valid
    if (!lpp::LppIsValidDefaultAgent(&lppAgent))
    {
        fprintf(
            stderr, "Couldn't create Live++ agent. Is LIVEPP_PATH correct?");
        return 1;
    }

    // enable Live++ for all loaded modules
    lppAgent.EnableModule(
        lpp::LppGetCurrentModulePath(),
        lpp::LPP_MODULES_OPTION_ALL_IMPORT_MODULES, nullptr, nullptr);
#endif // LIVEPP_PATH

    try
    {
        App::Settings settings = parseCli(argc, argv);

        App app{WHEELS_MOV(settings)};
        app.init();
        app.run();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception thrown: %s\n", e.what());
    }

#ifdef LIVEPP_PATH
    // destroy the Live++ agent
    lpp::LppDestroyDefaultAgent(&lppAgent);
#endif // LIVEPP_PATH

    return EXIT_SUCCESS;
}
