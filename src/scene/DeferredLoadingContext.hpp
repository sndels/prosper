#ifndef PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
#define PROSPER_SCENE_DEFERRED_LOADING_CONTEXT

#include "Allocators.hpp"
#include "Mesh.hpp"
#include "Texture.hpp"
#include "gfx/Fwd.hpp"
#include "gfx/Resources.hpp"
#include "scene/Fwd.hpp"
#include "scene/Material.hpp"
#include "utils/Timer.hpp"

#include <atomic>
#include <cgltf.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <mutex>
#include <thread>
#include <wheels/allocators/tlsf_allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/pair.hpp>
#include <wheels/containers/static_array.hpp>
#include <wheels/containers/string.hpp>

namespace scene
{

enum class IndicesType : uint8_t
{
    Uint8,
    Uint16,
    Uint32,
};

struct InputGeometryMetadata
{
    cgltf_accessor *indices{nullptr};
    cgltf_accessor *positions{nullptr};
    cgltf_accessor *normals{nullptr};
    cgltf_accessor *tangents{nullptr};
    cgltf_accessor *texCoord0s{nullptr};
    uint32_t sourceMeshIndex{0xFFFF'FFFF};
    uint32_t sourcePrimitiveIndex{0xFFFF'FFFF};
};

struct UploadedGeometryData
{
    shader_structs::GeometryMetadata metadata;
    uint32_t byteOffset{0};
    uint32_t byteCount{0};
    // This is valid while DeferredLoadingContext is
    wheels::StrSpan meshName;
};

gfx::Buffer createTextureStaging();

// Changes to this require changes to sMeshCacheVersion
struct MeshCacheHeader
{
    std::filesystem::file_time_type sourceWriteTime;
    uint32_t indexCount{0};
    uint32_t vertexCount{0};
    uint32_t meshletCount{0};
    // Offsets are for u32 values starting from the beginning of the blob.
    // The offset for indices is 0.
    uint32_t positionsOffset{0xFFFF'FFFF};
    uint32_t normalsOffset{0xFFFF'FFFF};
    uint32_t tangentsOffset{0xFFFF'FFFF};
    uint32_t texCoord0sOffset{0xFFFF'FFFF};
    uint32_t meshletsOffset{0xFFFF'FFFF};
    uint32_t meshletBoundsOffset{0xFFFF'FFFF};
    uint32_t meshletVerticesOffset{0xFFFF'FFFF};
    uint32_t meshletTrianglesByteOffset{0xFFFF'FFFF};
    uint32_t usesShortIndices{0};
    uint32_t blobByteCount{0};
};

class DeferredLoadingContext
{
  public:
    // Not noexcept because std::filesystem::path ctor
    DeferredLoadingContext() noexcept = default;
    ~DeferredLoadingContext();

    DeferredLoadingContext(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext(DeferredLoadingContext &&) = delete;
    DeferredLoadingContext &operator=(const DeferredLoadingContext &) = delete;
    DeferredLoadingContext &operator=(DeferredLoadingContext &&) = delete;

    void init(
        std::filesystem::path inSceneDir,
        std::filesystem::file_time_type inSceneWriteTime,
        cgltf_data &inGltfData);

    void launch();
    void kill();

    UploadedGeometryData uploadGeometryData(
        const MeshCacheHeader &cacheHeader,
        const wheels::Array<uint8_t> &dataBlob, const wheels::String &meshName);

    // TODO:
    // Make worker context private?
    // Make shared context private and access through methods that handle
    // mutexes?
    bool initialized{false};
    std::filesystem::path sceneDir;
    std::filesystem::file_time_type sceneWriteTime;
    // If there's no worker, main thread handles loading
    wheels::Optional<std::thread> worker;

    // Worker context
    cgltf_data *gltfData{nullptr};
    vk::CommandBuffer cb;
    uint32_t workerLoadedImageCount{0};
    wheels::HashSet<uint32_t> sRgbColorImages{gAllocators.loadingWorker};
    wheels::HashSet<uint32_t> linearColorImages{gAllocators.loadingWorker};
    wheels::Array<wheels::Pair<InputGeometryMetadata, MeshInfo>> meshes{
        gAllocators.loadingWorker};
    wheels::Array<wheels::String> meshNames{gAllocators.loadingWorker};
    gfx::Buffer geometryUploadBuffer;
    wheels::Array<uint32_t> geometryBufferRemainingByteCounts{
        gAllocators.loadingWorker};
    uint32_t workerLoadedMeshCount{0};
    utils::Timer meshTimer;
    utils::Timer textureTimer;

    // Shared context
    std::mutex geometryBuffersMutex;
    // These are only allocated by the async thread and it will not destroy
    // them. The managing thread is assumed to copy the buffers from here and is
    // responsible for their destruction. This array should only be read from
    // outside this class, write ops are not allowed.
    wheels::Array<gfx::Buffer> geometryBuffers{gAllocators.loadingWorker};
    std::mutex loadedMeshesMutex;
    wheels::Array<wheels::Pair<UploadedGeometryData, MeshInfo>> loadedMeshes{
        gAllocators.loadingWorker};

    std::mutex loadedTexturesMutex;
    wheels::Array<Texture2D> loadedTextures{gAllocators.loadingWorker};

    std::atomic<bool> interruptLoading{false};

    // Main context
    uint32_t geometryGeneration{0};
    uint32_t materialsGeneration{0};
    uint32_t framesSinceFinish{0};
    uint32_t textureArrayBinding{0};
    uint32_t loadedMeshCount{0};
    uint32_t loadedImageCount{0};
    uint32_t loadedMaterialCount{0};
    wheels::Array<shader_structs::MaterialData> materials{
        gAllocators.loadingWorker};
    gfx::Buffer stagingBuffer;

  private:
    uint32_t getGeometryBuffer(uint32_t byteCount);
};

} // namespace scene

#endif // PROSPER_SCENE_DEFERRED_LOADING_CONTEXT
