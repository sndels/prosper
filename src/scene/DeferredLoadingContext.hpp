#ifndef PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
#define PROSPER_SCENE_DEFERRED_LOADING_CONTEXT

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../utils/Timer.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"
#include "Material.hpp"
#include "Texture.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include <tiny_gltf.h>
#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/static_array.hpp>

Buffer createTextureStaging(Device *device);

struct DeferredLoadingContext
{
    DeferredLoadingContext(
        wheels::Allocator &alloc, Device *device,
        const std::filesystem::path *sceneDir,
        const tinygltf::Model &gltfModel);
    ~DeferredLoadingContext();

    DeferredLoadingContext(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext(DeferredLoadingContext &&) = delete;
    DeferredLoadingContext &operator=(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext &operator=(DeferredLoadingContext &&) = delete;

    Device *device{nullptr};
    // If there's no worker, main thread handles loading
    wheels::Optional<std::thread> worker;

    // Worker context
    tinygltf::Model gltfModel;
    vk::CommandBuffer cb;
    uint32_t workerLoadedImageCount{0};

    // Shared context
    std::mutex loadedTextureMutex;
    std::condition_variable loadedTextureTaken;
    wheels::Optional<Texture2D> loadedTexture;
    std::atomic<bool> interruptLoading{false};
    std::atomic<uint32_t> allocationHighWatermark{0};

    // Main context
    uint32_t materialsGeneration{0};
    uint32_t framesSinceFinish{0};
    uint32_t textureArrayBinding{0};
    uint32_t loadedImageCount{0};
    uint32_t loadedMaterialCount{0};
    wheels::Array<Material> materials;
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> stagingBuffers;
    Timer timer;
};

#endif // PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
