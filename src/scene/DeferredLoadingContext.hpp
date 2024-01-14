#ifndef PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
#define PROSPER_SCENE_DEFERRED_LOADING_CONTEXT

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../utils/Timer.hpp"
#include "../utils/Utils.hpp"
#include "Fwd.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "Texture.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <glm/glm.hpp>
#include <mutex>
#include <thread>
#include <tiny_gltf.h>
#include <wheels/allocators/tlsf_allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/pair.hpp>
#include <wheels/containers/static_array.hpp>

struct InputBuffer
{
    uint32_t index{0xFFFFFFFF};
    uint32_t byteOffset{0};
    uint32_t byteCount{0};
};

enum class IndicesType
{
    Uint8,
    Uint16,
    Uint32,
};

struct InputGeometryMetadata
{
    InputBuffer indices;
    InputBuffer positions;
    InputBuffer normals;
    InputBuffer tangents;
    InputBuffer texCoord0s;
    uint8_t indexByteWidth{0};
    uint32_t sourceMeshIndex{0xFFFFFFFF};
    uint32_t sourcePrimitiveIndex{0xFFFFFFFF};
};

struct UploadedGeometryData
{
    GeometryMetadata metadata;
    uint32_t byteCount{0};
};

Buffer createTextureStaging(Device *device);

// Changes to this require changes to sMeshCacheVersion
struct MeshCacheHeader
{
    std::filesystem::file_time_type sourceWriteTime;
    uint32_t indexCount{0};
    uint32_t vertexCount{0};
    // Offsets are for u32 values starting from the beginning of the blob.
    // The offset for indices is 0.
    uint32_t positionsOffset{0xFFFFFFFF};
    uint32_t normalsOffset{0xFFFFFFFF};
    uint32_t tangentsOffset{0xFFFFFFFF};
    uint32_t texCoord0sOffset{0xFFFFFFFF};
    uint32_t usesShortIndices{0};
    uint32_t blobByteCount{0};
};

class DeferredLoadingContext
{
  public:
    DeferredLoadingContext(
        Device *device, std::filesystem::path sceneDir,
        std::filesystem::file_time_type sceneWriteTime,
        const tinygltf::Model &gltfModel);
    ~DeferredLoadingContext();

    DeferredLoadingContext(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext(DeferredLoadingContext &&) = delete;
    DeferredLoadingContext &operator=(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext &operator=(DeferredLoadingContext &&) = delete;

    void launch();

    UploadedGeometryData uploadGeometryData(
        const MeshCacheHeader &cacheHeader,
        const wheels::Array<uint8_t> &dataBlob);

    // TODO:
    // Make worker context private?
    // Make shared context private and access through methods that handle
    // mutexes?
    Device *device{nullptr};
    std::filesystem::path sceneDir;
    std::filesystem::file_time_type sceneWriteTime;
    // If there's no worker, main thread handles loading
    wheels::Optional<std::thread> worker;

    // Worker context
    wheels::TlsfAllocator alloc;
    tinygltf::Model gltfModel;
    vk::CommandBuffer cb;
    uint32_t workerLoadedImageCount{0};
    wheels::Array<wheels::Pair<InputGeometryMetadata, MeshInfo>> meshes;
    Buffer geometryUploadBuffer;
    wheels::Array<uint32_t> geometryBufferRemainingByteCounts{alloc};
    uint32_t workerLoadedMeshCount{0};
    Timer meshTimer;
    Timer textureTimer;

    // Shared context
    std::mutex geometryBuffersMutex;
    // These are only allocated by the async thread and it will not destroy
    // them. The managing thread is assumed to copy the buffers from here and is
    // responsible for their destruction. This array should only be read from
    // outside this class, write ops are not allowed.
    wheels::Array<Buffer> geometryBuffers{alloc};
    std::mutex loadedMeshesMutex;
    wheels::Array<wheels::Pair<UploadedGeometryData, MeshInfo>> loadedMeshes;

    std::mutex loadedTexturesMutex;
    wheels::Array<Texture2D> loadedTextures;

    std::atomic<bool> interruptLoading{false};
    std::atomic<uint32_t> linearAllocatorHighWatermark{0};
    std::atomic<uint32_t> generalAllocatorHightWatermark{0};

    // Main context
    uint32_t geometryGeneration{0};
    uint32_t materialsGeneration{0};
    uint32_t framesSinceFinish{0};
    uint32_t textureArrayBinding{0};
    uint32_t loadedMeshCount{0};
    uint32_t loadedImageCount{0};
    uint32_t loadedMaterialCount{0};
    wheels::Array<Material> materials{alloc};
    wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> stagingBuffers;

  private:
    uint32_t getGeometryBuffer(uint32_t byteCount);
};

#endif // PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
