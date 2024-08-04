#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cxxopts.hpp>
#include <tomlcpp.hpp>

#include "Allocators.hpp"
#include "App.hpp"
#include "Window.hpp"
#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "render/RenderResources.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
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

const char *const sConfigArg = "config";
// These can be given in the config TOML as root level key-values
const char *const sDebugLayersArg = "debugLayers";                     // bool
const char *const sShaderDisassemblyArg = "dumpShaderDisassembly";     // bool
const char *const sBreakOnValidationErrArg = "breakOnValidationError"; // bool
const char *const sRobustAccessArg = "robustAccess";                   // bool
const char *const sSceneFileArg = "sceneFile"; // string, path

// NOLINTNEXTLINE(*-avoid-c-arrays): Mandatory
App::Settings parseCli(int argc, char *argv[])
{
    cxxopts::Options options("prosper", "A toy Vulkan renderer");
    // clang-format off
        options.add_options()
            (sConfigArg, "Config file to use. Any CLI flags take precedence.",
             cxxopts::value<std::string>()->default_value(""))
            (sDebugLayersArg, "Enable Vulkan debug layers")
            (sShaderDisassemblyArg, "Dump shader disassembly to info log")
            (sBreakOnValidationErrArg, "Break debugger on Vulkan validation error")
            (sRobustAccessArg, "Enable VK_EXT_robustness2 for buffers and images")
            (sSceneFileArg, std::string{"Scene to open (default: '"} + s_default_scene_path +"')",
             cxxopts::value<std::string>()->default_value(""));
    // clang-format on
    options.parse_positional({"sceneFile"});
    const cxxopts::ParseResult args = options.parse(argc, argv);

    std::filesystem::path scenePath;
    Device::Settings deviceSettings;

    // Try to parse toml first as we'll override any of its settings with values
    // given in the CLI
    const std::filesystem::path configPath = args["config"].as<std::string>();
    if (!configPath.empty() && std::filesystem::is_regular_file(configPath))
    {
        const toml::Result result = toml::parseFile(configPath.string());
        if (result.table == nullptr)
            LOG_ERR(
                "Couldn't parse config from '%s': %s",
                configPath.string().c_str(), result.errmsg.c_str());
        else
        {
            {
                auto [ok, path] = result.table->getString(sSceneFileArg);
                if (ok)
                    scenePath = path;
            }

            const auto tryGetFlag = [&result](bool &dst, const char *name)
            {
                auto [ok, flag] = result.table->getBool(name);
                if (ok)
                    dst = flag;
            };

            tryGetFlag(deviceSettings.enableDebugLayers, sDebugLayersArg);
            tryGetFlag(
                deviceSettings.dumpShaderDisassembly, sShaderDisassemblyArg);
            tryGetFlag(
                deviceSettings.breakOnValidationError,
                sBreakOnValidationErrArg);
            tryGetFlag(deviceSettings.robustAccess, sRobustAccessArg);
        }
    }

    // Parse explicit CLI flags after TOML to give them precedence
    if (args.count(sSceneFileArg) > 0)
        scenePath = args[sSceneFileArg].as<std::string>();

    {
        const auto tryGetFlag = [&args](bool &dst, const char *name)
        {
            if (args.count(name) > 0)
                dst = args[name].as<bool>();
        };
        tryGetFlag(deviceSettings.enableDebugLayers, sDebugLayersArg);
        tryGetFlag(deviceSettings.dumpShaderDisassembly, sShaderDisassemblyArg);
        tryGetFlag(
            deviceSettings.breakOnValidationError, sBreakOnValidationErrArg);
        tryGetFlag(deviceSettings.robustAccess, sRobustAccessArg);
    }

    if (scenePath.empty())
        scenePath = s_default_scene_path;

    return App::Settings{
        .scene = scenePath,
        .device = deviceSettings,
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
    lpp::LppDefaultAgent lppAgent =
        lpp::LppCreateDefaultAgent(nullptr, LIVEPP_PATH);
    if (!lpp::LppIsValidDefaultAgent(&lppAgent))
    {
        LOG_ERR("Couldn't create Live++ agent. Is LIVEPP_PATH correct?");
        return 1;
    }

    lppAgent.EnableModule(
        lpp::LppGetCurrentModulePath(),
        lpp::LPP_MODULES_OPTION_ALL_IMPORT_MODULES, nullptr, nullptr);
#endif // LIVEPP_PATH

    setCurrentThreadName("prosper main");

    try
    {
        App::Settings settings = parseCli(argc, argv);

        // Used by environment map KTX loading so conservative
        LinearAllocator scratchBacking{megabytes(128)};
        ScopedScratch scopeAlloc{scratchBacking};

        const auto &tl = [](const char *stage, std::function<void()> const &fn)
        {
            const Timer t;
            fn();
            LOG_INFO("%s took %.2fs", stage, t.getSeconds());
        };

        // Globals
        // Only one of these exist, and passing them around or storing pointers
        // to them in classes adds needless noise. This style of global avoids
        // many issues in initialization order. More in Game Engine Architecture
        // 3rd ed. section 6.1.2
        tl("Allocators init", []() { gAllocators.init(); });
        defer { gAllocators.destroy(); };

        // gInputHandler doesn't require calling init
        tl("Window init", []() { gWindow.init(sStartupRes, sWindowTitle); });
        defer { gWindow.destroy(); };

        tl("Device init", [&scopeAlloc, &settings]()
           { gDevice.init(scopeAlloc.child_scope(), settings.device); });
        defer { gDevice.destroy(); };

        gRenderResources.init();
        defer { gRenderResources.destroy(); };

        gStaticDescriptorsAlloc.init();
        defer { gStaticDescriptorsAlloc.destroy(); };

        gProfiler.init();
        defer { gProfiler.destroy(); };

        App app{settings.scene};
        app.init(WHEELS_MOV(scopeAlloc));

        app.setInitScratchHighWatermark(
            scratchBacking.allocated_byte_count_high_watermark());
        // We don't need this memory anymore so let's drop it.
        scratchBacking.destroy();

        LOG_INFO("run() called after %.2fs", t.getSeconds());
        app.run();
    }
    catch (std::exception &e)
    {
        LOG_ERR("Exception thrown: %s", e.what());
    }

#ifdef LIVEPP_PATH
    lpp::LppDestroyDefaultAgent(&lppAgent);
#endif // LIVEPP_PATH

    return EXIT_SUCCESS;
}
