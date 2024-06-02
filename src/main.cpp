#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cxxopts.hpp>

#include "Allocators.hpp"
#include "App.hpp"
#include "Window.hpp"
#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "utils/Utils.hpp"

#ifdef LIVEPP_PATH
#include "API/x64/LPP_API_x64_CPP.h"
#endif // LIVEPP_PATH

#ifdef _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif // _CRTDBG_MAP_ALLOC

using namespace wheels;

namespace
{

const Pair<uint32_t, uint32_t> sStartupRes{1920u, 1080u};
const char *const sWindowTitle = "prosper";

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
#ifdef _CRTDBG_MAP_ALLOC
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    // Set to the allocation index given by the dump
    // _CrtSetBreakAlloc(0);
#endif // _CRTDBG_MAP_ALLOC

    const Timer t;

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

        LinearAllocator scratchBacking{megabytes(16)};
        ScopedScratch scopeAlloc{scratchBacking};

        const auto &tl = [](const char *stage, std::function<void()> const &fn)
        {
            const Timer t;
            fn();
            printf("%s took %.2fs\n", stage, t.getSeconds());
        };

        // Globals
        // Only one of these exist, and passing them around or storing pointers
        // to them in classes adds needless noise. This style of global avoids
        // many issues in initialization order. More in Game Engine Architecture
        // 3rd ed. section 6.1.2
        tl("Allocators init", []() { gAllocators.init(); });
        // gInputHandler doesn't require calling init
        tl("Window init", []() { gWindow.init(sStartupRes, sWindowTitle); });
        tl("Device init", [&scopeAlloc, &settings]()
           { gDevice.init(scopeAlloc.child_scope(), settings.device); });

        gRenderResources.init();

        App app{settings.scene};
        app.init(WHEELS_MOV(scopeAlloc));
        app.setInitScratchHighWatermark(
            scratchBacking.allocated_byte_count_high_watermark());
        printf("run() called after %.2fs\n", t.getSeconds());
        app.run();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception thrown: %s\n", e.what());
    }

    gRenderResources.destroy();
    gDevice.destroy();
    gWindow.destroy();
    gAllocators.destroy();

#ifdef LIVEPP_PATH
    // destroy the Live++ agent
    lpp::LppDestroyDefaultAgent(&lppAgent);
#endif // LIVEPP_PATH

    return EXIT_SUCCESS;
}
