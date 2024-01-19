#include "DeferredLoadingContext.hpp"

#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <wheels/allocators/linear_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sGeometryBufferSize = asserted_cast<uint32_t>(megabytes(64));

template <typename T>
void copyInputData(
    Array<T> &dst, const tinygltf::Model &gltfModel,
    const InputBuffer &srcBuffer)
{
    if (dst.size() == 0)
        return;

    const tinygltf::Buffer &gltfBuffer = gltfModel.buffers[srcBuffer.index];
    const size_t byteCount = dst.size() * sizeof(T);
    WHEELS_ASSERT(byteCount == srcBuffer.byteCount);

    memcpy(
        dst.data(), gltfBuffer.data.data() + srcBuffer.byteOffset, byteCount);
}

template <typename T>
void remapVertexAttribute(
    Allocator &alloc, Array<T> &src, const Array<uint32_t> &remapIndices,
    size_t uniqueVertexCount)
{
    Array<T> remapped{alloc};
    remapped.resize(uniqueVertexCount);
    meshopt_remapVertexBuffer(
        remapped.data(), src.data(), src.size(), sizeof(T),
        remapIndices.data());

    src = WHEELS_MOV(remapped);
}

struct MeshData
{
    wheels::Array<uint32_t> indices;
    wheels::Array<glm::vec3> positions;
    wheels::Array<glm::vec3> normals;
    wheels::Array<glm::vec4> tangents;
    wheels::Array<glm::vec2> texCoord0s;
};

MeshData getMeshData(
    Allocator &alloc, const tinygltf::Model &gltfModel,
    const InputGeometryMetadata &metadata, const MeshInfo &meshInfo)
{
    MeshData ret{
        .indices = Array<uint32_t>{alloc},
        .positions = Array<vec3>{alloc},
        .normals = Array<vec3>{alloc},
        .tangents = Array<vec4>{alloc},
        .texCoord0s = Array<vec2>{alloc},
    };

    ret.indices.resize(meshInfo.indexCount);
    if (metadata.indexByteWidth == sizeof(uint8_t))
    {
        const tinygltf::Buffer &gltfBuffer =
            gltfModel.buffers[metadata.indices.index];
        WHEELS_ASSERT(
            sizeof(uint8_t) * meshInfo.indexCount ==
            metadata.indices.byteCount);

        const uint8_t *src = reinterpret_cast<const uint8_t *>(
            gltfBuffer.data.data() + metadata.indices.byteOffset);
        for (uint32_t i = 0; i < meshInfo.indexCount; ++i)
            ret.indices[i] = static_cast<uint32_t>(src[i]);
    }
    else if (metadata.indexByteWidth == sizeof(uint16_t))
    {
        const tinygltf::Buffer &gltfBuffer =
            gltfModel.buffers[metadata.indices.index];
        // Don't fail if there's padding in source data
        WHEELS_ASSERT(
            sizeof(uint16_t) * meshInfo.indexCount ==
                metadata.indices.byteCount ||
            sizeof(uint16_t) * (meshInfo.indexCount + 1) ==
                metadata.indices.byteCount);

        const uint16_t *src = reinterpret_cast<const uint16_t *>(
            gltfBuffer.data.data() + metadata.indices.byteOffset);
        for (uint32_t i = 0; i < meshInfo.indexCount; ++i)
            ret.indices[i] = static_cast<uint32_t>(src[i]);
    }
    else
    {
        WHEELS_ASSERT(metadata.indexByteWidth == sizeof(uint32_t));
        copyInputData(ret.indices, gltfModel, metadata.indices);
    }

    ret.positions.resize(meshInfo.vertexCount);
    copyInputData(ret.positions, gltfModel, metadata.positions);

    ret.normals.resize(meshInfo.vertexCount);
    copyInputData(ret.normals, gltfModel, metadata.normals);

    const bool hasTangents = metadata.tangents.index < 0xFFFFFFFF;
    if (hasTangents)
    {
        ret.tangents.resize(meshInfo.vertexCount);
        copyInputData(ret.tangents, gltfModel, metadata.tangents);
    }

    const bool hasTexCoord0s = metadata.texCoord0s.index < 0xFFFFFFFF;
    if (hasTexCoord0s)
    {
        ret.texCoord0s.resize(meshInfo.vertexCount);
        copyInputData(ret.texCoord0s, gltfModel, metadata.texCoord0s);
    }

    return ret;
}

// mikktspace defined interface
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,bugprone-easily-swappable-parameters)

int mikkTGetNumFaces(const SMikkTSpaceContext *pContext)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    const MeshData *meshData =
        reinterpret_cast<const MeshData *>(pContext->m_pUserData);

    WHEELS_ASSERT(
        meshData->positions.size() % 3 == 0 &&
        "We assume only tris in the data");

    const size_t faceCount = meshData->positions.size() / 3;

    return asserted_cast<int>(faceCount);
}

int mikkTGetNumVerticesOfFace(
    const SMikkTSpaceContext * /*pContext*/, const int /*iFace*/)
{
    // We only have tris
    return 3;
}

int mikkTVertexIndex(const int iFace, const int iVert)
{
    // Go through faces in the opposite order, this seems to fix the glTF normal
    // map handedness problem in glTF2.0 NormalTangentTest
    return iFace * 3 + (2 - iVert);
}

void mikkTGetPosition(
    const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace,
    const int iVert)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    const MeshData *meshData =
        reinterpret_cast<const MeshData *>(pContext->m_pUserData);

    const int vertexI = mikkTVertexIndex(iFace, iVert);
    const vec3 &pos = meshData->positions[vertexI];

    memcpy(fvPosOut, &pos.x, sizeof(pos));
}

void mikkTGetNormal(
    const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace,
    const int iVert)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    const MeshData *meshData =
        reinterpret_cast<const MeshData *>(pContext->m_pUserData);

    const int vertexI = mikkTVertexIndex(iFace, iVert);
    const vec3 &normal = meshData->normals[vertexI];

    memcpy(fvNormOut, &normal.x, sizeof(normal));
}

void mikkTGetTexCoord(
    const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace,
    const int iVert)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    const MeshData *meshData =
        reinterpret_cast<const MeshData *>(pContext->m_pUserData);

    const int vertexI = mikkTVertexIndex(iFace, iVert);
    const vec2 &texCoord0 = meshData->texCoord0s[vertexI];

    memcpy(fvTexcOut, &texCoord0.x, sizeof(texCoord0));
}

void mikkTSetTSpaceBasic(
    const SMikkTSpaceContext *pContext, const float fvTangent[],
    const float fSign, const int iFace, const int iVert)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    MeshData *meshData = reinterpret_cast<MeshData *>(pContext->m_pUserData);

    const int vertexI = mikkTVertexIndex(iFace, iVert);
    meshData->tangents[vertexI] =
        vec4{fvTangent[0], fvTangent[1], fvTangent[2], fSign};
}

// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,bugprone-easily-swappable-parameters)

template <typename T>
void flattenAttribute(
    Allocator &alloc, Array<T> &attribute, const Array<uint32_t> &indices)
{
    Array<T> flattened{alloc, indices.size()};
    for (const uint32_t i : indices)
        flattened.push_back(attribute[i]);

    attribute = WHEELS_MOV(flattened);
}

// alloc needs to be the same as the one used for meshData
void generateTangents(Allocator &alloc, MeshData *meshData)
{
    WHEELS_ASSERT(meshData != nullptr);
    WHEELS_ASSERT(meshData->tangents.empty());
    WHEELS_ASSERT(meshData->positions.size() == meshData->normals.size());
    WHEELS_ASSERT(meshData->positions.size() == meshData->texCoord0s.size());

    // Flatten data first as instructed in the mikktspace header
    // TODO: tmp buffers here
    const size_t flattenedVertexCount = meshData->indices.size();
    flattenAttribute(alloc, meshData->positions, meshData->indices);
    flattenAttribute(alloc, meshData->normals, meshData->indices);
    flattenAttribute(alloc, meshData->texCoord0s, meshData->indices);
    meshData->indices.clear();

    meshData->tangents.resize(flattenedVertexCount);

    // Now we can generate the tangents
    SMikkTSpaceInterface sMikkTInterface{
        .m_getNumFaces = &mikkTGetNumFaces,
        .m_getNumVerticesOfFace = mikkTGetNumVerticesOfFace,
        .m_getPosition = mikkTGetPosition,
        .m_getNormal = mikkTGetNormal,
        .m_getTexCoord = mikkTGetTexCoord,
        .m_setTSpaceBasic = mikkTSetTSpaceBasic,
    };

    const SMikkTSpaceContext mikkCtx{
        .m_pInterface = &sMikkTInterface,
        .m_pUserData = meshData,
    };

    genTangSpaceDefault(&mikkCtx);

    // And now that we have tangents, we can re-generate indices
    const StaticArray vertexStreams{{
        meshopt_Stream{
            .data = meshData->positions.data(),
            .size = sizeof(vec3),
            .stride = sizeof(vec3),
        },
        meshopt_Stream{
            .data = meshData->normals.data(),
            .size = sizeof(vec3),
            .stride = sizeof(vec3),
        },
        meshopt_Stream{
            .data = meshData->tangents.data(),
            .size = sizeof(vec4),
            .stride = sizeof(vec4),
        },
        meshopt_Stream{
            .data = meshData->texCoord0s.data(),
            .size = sizeof(vec2),
            .stride = sizeof(vec2),
        },
    }};

    Array<uint32_t> remapTable{alloc};
    remapTable.resize(flattenedVertexCount);
    const size_t uniqueVertexCount = meshopt_generateVertexRemapMulti(
        remapTable.data(), nullptr, flattenedVertexCount, flattenedVertexCount,
        vertexStreams.data(), vertexStreams.size());

    meshData->indices.resize(flattenedVertexCount);
    meshopt_remapIndexBuffer(
        meshData->indices.data(), nullptr, flattenedVertexCount,
        remapTable.data());

    remapVertexAttribute(
        alloc, meshData->positions, remapTable, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->normals, remapTable, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->tangents, remapTable, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->texCoord0s, remapTable, uniqueVertexCount);
}

// alloc needs to be the same as the one used for meshData
void optimizeMeshData(
    Allocator &alloc, MeshData *meshData, const std::string &meshName)
{
    WHEELS_ASSERT(meshData != nullptr);

    const size_t indexCount = meshData->indices.size();
    const size_t vertexCount = meshData->positions.size();

    Array<uint32_t> tmpIndices{alloc};
    tmpIndices.resize(meshData->indices.size());
    meshopt_optimizeVertexCache(
        tmpIndices.data(), meshData->indices.data(), indexCount, vertexCount);

    const float vertexCacheDegradationThreshod = 1.00f;
    meshopt_optimizeOverdraw(
        meshData->indices.data(), tmpIndices.data(), tmpIndices.size(),
        &meshData->positions.data()[0].x, vertexCount, sizeof(vec3),
        vertexCacheDegradationThreshod);

    Array<uint32_t> remapIndices{alloc};
    remapIndices.resize(vertexCount);
    const size_t uniqueVertexCount = meshopt_optimizeVertexFetchRemap(
        remapIndices.data(), meshData->indices.data(), indexCount, vertexCount);
    if (uniqueVertexCount < vertexCount)
        fprintf(stderr, "Mesh '%s' has unused vertices\n", meshName.c_str());

    // Reuse tmpIndices as it's not required after optimizeOverdraw
    meshopt_remapIndexBuffer(
        tmpIndices.data(), meshData->indices.data(), indexCount,
        remapIndices.data());
    meshData->indices = WHEELS_MOV(tmpIndices);

    remapVertexAttribute(
        alloc, meshData->positions, remapIndices, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->normals, remapIndices, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->tangents, remapIndices, uniqueVertexCount);
    remapVertexAttribute(
        alloc, meshData->texCoord0s, remapIndices, uniqueVertexCount);
}

const uint64_t sMeshCacheMagic = 0x48534D5250535250; // PRSPRMSH
// This should be incremented when breaking changes are made to
// what's cached
const uint32_t sMeshCacheVersion = 1;

std::filesystem::path getCachePath(
    const std::filesystem::path &sceneDir, uint32_t meshIndex)
{
    const std::string filename =
        "cache" + std::to_string(meshIndex) + ".prosper_mesh";
    std::filesystem::path ret{sceneDir / "prosper_cache" / filename};
    return ret;
}

// Returns the read header or null if cache wasn't up to date. If a pointer to
// an array for the data blob is given, its resized and the blob is read into
// it.
Optional<MeshCacheHeader> readCache(
    const std::filesystem::path &cachePath,
    Array<uint8_t> *dataBlobOut = nullptr)
{
    Optional<MeshCacheHeader> ret;

    if (!std::filesystem::exists(cachePath))
    {
        fprintf(stdout, "Missing cache for %s\n", cachePath.string().c_str());
        return ret;
    }

    std::ifstream cacheFile{cachePath, std::ios_base::binary};

    uint64_t magic{0};
    static_assert(sizeof(magic) == sizeof(sMeshCacheMagic));

    readRaw(cacheFile, magic);
    if (magic != sMeshCacheMagic)
        throw std::runtime_error(
            "Expected a valid mesh cache in file '" + cachePath.string() + "'");

    uint32_t version{0};
    static_assert(sizeof(version) == sizeof(sMeshCacheVersion));
    readRaw(cacheFile, version);
    if (sMeshCacheVersion != version)
    {
        fprintf(
            stdout, "Old cache data version for %s\n",
            cachePath.string().c_str());
        return ret;
    }

    ret = MeshCacheHeader{};

    readRaw(cacheFile, ret->sourceWriteTime);
    readRaw(cacheFile, ret->indexCount);
    readRaw(cacheFile, ret->vertexCount);
    readRaw(cacheFile, ret->positionsOffset);
    readRaw(cacheFile, ret->normalsOffset);
    readRaw(cacheFile, ret->tangentsOffset);
    readRaw(cacheFile, ret->texCoord0sOffset);
    readRaw(cacheFile, ret->usesShortIndices);
    readRaw(cacheFile, ret->blobByteCount);

    if (dataBlobOut != nullptr)
    {
        dataBlobOut->resize(ret->blobByteCount);
        readRawSpan(cacheFile, Span{dataBlobOut->data(), dataBlobOut->size()});
    }

    return ret;
}

bool cacheValid(
    const std::filesystem::path &cachePath,
    std::filesystem::file_time_type sceneWriteTime)
{
    const Optional<MeshCacheHeader> header = readCache(cachePath);
    if (!header.has_value())
        return false;

    if (header->sourceWriteTime != sceneWriteTime)
    {
        fprintf(stdout, "Stale cache for %s\n", cachePath.string().c_str());
        return false;
    }
    return true;
}

void writeCache(
    ScopedScratch scopeAlloc, const std::filesystem::path &sceneDir,
    std::filesystem::file_time_type sceneWriteTime, uint32_t meshIndex,
    MeshData &&meshData, const MeshInfo &meshInfo)
{
    WHEELS_ASSERT(meshData.indices.size() == meshInfo.indexCount);
    WHEELS_ASSERT(meshData.positions.size() == meshInfo.vertexCount);
    WHEELS_ASSERT(meshData.normals.size() == meshInfo.vertexCount);
    WHEELS_ASSERT(
        meshData.tangents.size() == meshInfo.vertexCount ||
        meshData.tangents.empty());
    WHEELS_ASSERT(
        meshData.texCoord0s.size() == meshInfo.vertexCount ||
        meshData.texCoord0s.empty());

    const bool hasTangents = !meshData.tangents.empty();
    const bool hasTexCoord0s = !meshData.texCoord0s.empty();

    const bool usesShortIndices = meshInfo.vertexCount <= 0xFFFF;
    Array<uint8_t> packedIndices{scopeAlloc};
    if (usesShortIndices)
    {
        size_t byteCount = meshInfo.indexCount * sizeof(uint16_t);
        // Let's pad to 4byte boundary to make things simpler later
        byteCount = (byteCount + sizeof(uint32_t) - 1) / sizeof(uint32_t) *
                    sizeof(uint32_t);
        WHEELS_ASSERT(byteCount % sizeof(uint32_t) == 0);

        packedIndices.resize(byteCount);
        uint16_t *dst = reinterpret_cast<uint16_t *>(packedIndices.data());
        for (size_t i = 0; i < meshInfo.indexCount; ++i)
            dst[i] = asserted_cast<uint16_t>(meshData.indices[i]);
    }
    else
    {
        // TODO:
        // Just use the data directly instead the extra allocation and
        // memcpy?
        const size_t byteCount = meshInfo.indexCount * sizeof(uint32_t);
        packedIndices.resize(byteCount);
        memcpy(packedIndices.data(), meshData.indices.data(), byteCount);
    }

    // Figure out the required storage
    const uint32_t indicesByteCount =
        asserted_cast<uint32_t>(packedIndices.size());
    const uint32_t positionsByteCount =
        asserted_cast<uint32_t>(meshData.positions.size() * sizeof(vec3));
    const uint32_t normalsByteCount =
        asserted_cast<uint32_t>(meshData.normals.size() * sizeof(vec3));
    const uint32_t tangentsByteCount =
        asserted_cast<uint32_t>(meshData.tangents.size() * sizeof(vec4));
    const uint32_t texCoord0sByteCount =
        asserted_cast<uint32_t>(meshData.texCoord0s.size() * sizeof(vec2));
    const uint32_t byteCount = indicesByteCount + positionsByteCount +
                               normalsByteCount + tangentsByteCount +
                               texCoord0sByteCount;
    WHEELS_ASSERT(
        byteCount < sGeometryBufferSize &&
        "The default size for geometry buffers doesn't fit the mesh");

    // Offsets into GPU buffer are for uint
    const uint32_t elementSize = static_cast<uint32_t>(sizeof(uint32_t));

    const MeshCacheHeader header{
        .sourceWriteTime = sceneWriteTime,
        .indexCount = meshInfo.indexCount,
        .vertexCount = meshInfo.vertexCount,
        .positionsOffset = indicesByteCount / elementSize,
        .normalsOffset = (indicesByteCount + positionsByteCount) / elementSize,
        .tangentsOffset = hasTangents ? (indicesByteCount + positionsByteCount +
                                         normalsByteCount) /
                                            elementSize
                                      : 0xFFFFFFFF,
        .texCoord0sOffset = hasTexCoord0s
                                ? (indicesByteCount + positionsByteCount +
                                   normalsByteCount + tangentsByteCount) /
                                      elementSize
                                : 0xFFFFFFFF,
        .usesShortIndices = usesShortIndices ? 1u : 0u,
        .blobByteCount = byteCount,
    };

    const std::filesystem::path cachePath = getCachePath(sceneDir, meshIndex);

    const std::filesystem::path cacheFolder = cachePath.parent_path();
    if (!std::filesystem::exists(cacheFolder))
        std::filesystem::create_directories(cacheFolder);

    std::filesystem::remove(cachePath);

    // Write into a tmp file and rename when done to minimize the potential for
    // corrupted files
    std::filesystem::path cacheTmpPath = cachePath;
    cacheTmpPath.replace_extension("prosper_mesh_TMP");

    // NOTE:
    // Caches aren't supposed to be portable so we don't pay attention to
    // endianness.
    std::ofstream cacheFile{cacheTmpPath, std::ios_base::binary};
    writeRaw(cacheFile, sMeshCacheMagic);
    writeRaw(cacheFile, sMeshCacheVersion);

    writeRaw(cacheFile, header.sourceWriteTime);
    writeRaw(cacheFile, header.indexCount);
    writeRaw(cacheFile, header.vertexCount);
    writeRaw(cacheFile, header.positionsOffset);
    writeRaw(cacheFile, header.normalsOffset);
    writeRaw(cacheFile, header.tangentsOffset);
    writeRaw(cacheFile, header.texCoord0sOffset);
    writeRaw(cacheFile, header.usesShortIndices);
    writeRaw(cacheFile, header.blobByteCount);

    const std::streampos blobStart = cacheFile.tellp();
    // TODO:
    // Wheels: Array::span() defaults that return a full span
    writeRawSpan(cacheFile, Span{packedIndices.data(), packedIndices.size()});
    writeRawSpan(
        cacheFile, Span{meshData.positions.data(), meshData.positions.size()});
    writeRawSpan(
        cacheFile, Span{meshData.normals.data(), meshData.normals.size()});
    writeRawSpan(
        cacheFile, Span{meshData.tangents.data(), meshData.tangents.size()});
    writeRawSpan(
        cacheFile,
        Span{meshData.texCoord0s.data(), meshData.texCoord0s.size()});
    const std::streampos blobEnd = cacheFile.tellp();
    const std::streamoff blobLen = blobEnd - blobStart;
    WHEELS_ASSERT(blobLen == header.blobByteCount);

    cacheFile.close();

    // Make sure we have rw permissions for the user to be nice
    const std::filesystem::perms initialPerms =
        std::filesystem::status(cacheTmpPath).permissions();
    std::filesystem::permissions(
        cacheTmpPath, initialPerms | std::filesystem::perms::owner_read |
                          std::filesystem::perms::owner_write);

    // Rename when the file is done to minimize the potential of a corrupted
    // file
    std::filesystem::rename(cacheTmpPath, cachePath);
}

void loadNextMesh(ScopedScratch scopeAlloc, DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);

    const uint32_t meshIndex = ctx->workerLoadedMeshCount;
    WHEELS_ASSERT(meshIndex < ctx->meshes.size());

    // Ctx member functions will use the command buffer
    ctx->cb.reset();
    ctx->cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    const QueueFamilies &families = ctx->device->queueFamilies();
    WHEELS_ASSERT(families.graphicsFamily.has_value());
    WHEELS_ASSERT(families.transferFamily.has_value());

    const Pair<InputGeometryMetadata, MeshInfo> &nextMesh =
        ctx->meshes[meshIndex];
    const InputGeometryMetadata &metadata = nextMesh.first;
    MeshInfo info = nextMesh.second;

    const std::filesystem::path cachePath =
        getCachePath(ctx->sceneDir, meshIndex);
    if (!cacheValid(cachePath, ctx->sceneWriteTime))
    {
        MeshData meshData =
            getMeshData(scopeAlloc, ctx->gltfModel, metadata, info);

        if (meshData.tangents.empty() && !meshData.texCoord0s.empty())
        {
            generateTangents(scopeAlloc, &meshData);
            info.vertexCount =
                asserted_cast<uint32_t>(meshData.positions.size());
        }

        optimizeMeshData(
            scopeAlloc, &meshData,
            ctx->gltfModel.meshes[metadata.sourceMeshIndex].name);

        writeCache(
            scopeAlloc.child_scope(), ctx->sceneDir, ctx->sceneWriteTime,
            meshIndex, WHEELS_MOV(meshData), info);
    }

    // Always read from the cache to make caching issues always visible
    Array<uint8_t> dataBlob{scopeAlloc};
    const Optional<MeshCacheHeader> cacheHeader =
        readCache(cachePath, &dataBlob);
    WHEELS_ASSERT(cacheHeader.has_value());
    WHEELS_ASSERT(cacheHeader->indexCount == info.indexCount);
    // Tangent generation can change vertex count
    info.vertexCount = cacheHeader->vertexCount;

    const UploadedGeometryData uploadData =
        ctx->uploadGeometryData(*cacheHeader, dataBlob);

    if (*families.graphicsFamily != *families.transferFamily)
    {
        const Buffer &buffer =
            ctx->geometryBuffers[uploadData.metadata.bufferIndex];

        // Transfer ownership of the newly pushed buffer range.
        // NOTE: This expects the subsequent ranges to be packed tightly.
        // Extra bytes in between should not happen since the buffer is
        // bound up to the final offset + bytecount.
        const vk::BufferMemoryBarrier2 releaseBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eNone,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
            .srcQueueFamilyIndex = *families.transferFamily,
            .dstQueueFamilyIndex = *families.graphicsFamily,
            .buffer = buffer.handle,
            .offset = uploadData.metadata.indicesOffset,
            .size = uploadData.byteCount,
        };
        ctx->cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &releaseBarrier,
        });
    }

    ctx->cb.end();

    const vk::Queue transferQueue = ctx->device->transferQueue();
    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cb,
    };
    checkSuccess(
        transferQueue.submit(1, &submitInfo, vk::Fence{}),
        "submitTextureUpload");
    // We could have multiple uploads in flight, but let's be simple for now
    transferQueue.waitIdle();

    ctx->workerLoadedMeshCount++;

    {
        const std::lock_guard _lock{ctx->loadedMeshesMutex};

        ctx->loadedMeshes.emplace_back(uploadData, info);
    }

    if (ctx->workerLoadedMeshCount == ctx->meshes.size())
    {
        printf("Mesh loading took %.2fs\n", ctx->meshTimer.getSeconds());
        ctx->textureTimer.reset();
    }
}

void loadNextTexture(ScopedScratch scopeAlloc, DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);

    if (ctx->workerLoadedImageCount == ctx->gltfModel.images.size())
    {
        printf("Texture loading took %.2fs\n", ctx->textureTimer.getSeconds());
        ctx->interruptLoading = true;
        return;
    }

    WHEELS_ASSERT(ctx->gltfModel.images.size() > ctx->workerLoadedImageCount);
    const tinygltf::Image &image =
        ctx->gltfModel.images[ctx->workerLoadedImageCount];
    if (image.uri.empty())
        throw std::runtime_error("Embedded glTF textures aren't supported. "
                                 "Scene should be glTF + "
                                 "bin + textures.");

    ctx->cb.reset();
    ctx->cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    Texture2D tex{
        scopeAlloc.child_scope(),
        ctx->device,
        ctx->sceneDir / image.uri,
        ctx->cb,
        ctx->stagingBuffers[0],
        true,
        true};

    const QueueFamilies &families = ctx->device->queueFamilies();
    WHEELS_ASSERT(families.graphicsFamily.has_value());
    WHEELS_ASSERT(families.transferFamily.has_value());

    if (*families.graphicsFamily != *families.transferFamily)
    {
        const vk::ImageMemoryBarrier2 releaseBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eNone,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = *families.transferFamily,
            .dstQueueFamilyIndex = *families.graphicsFamily,
            .image = tex.nativeHandle(),
            .subresourceRange =
                vk::ImageSubresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
        };
        ctx->cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &releaseBarrier,
        });
    }

    ctx->cb.end();

    const vk::Queue transferQueue = ctx->device->transferQueue();
    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cb,
    };
    checkSuccess(
        transferQueue.submit(1, &submitInfo, vk::Fence{}),
        "submitTextureUpload");
    // We could have multiple uploads in flight, but let's be simple for now
    transferQueue.waitIdle();

    ctx->workerLoadedImageCount++;

    {
        const std::lock_guard _lock{ctx->loadedTexturesMutex};

        ctx->loadedTextures.emplace_back(WHEELS_MOV(tex));
    }
}

void loadingWorker(DeferredLoadingContext *ctx)
{
    // TODO:
    // Make clang-tidy treat WHEELS_ASSERT as assert so that it considers
    // them valid null checks
    WHEELS_ASSERT(
        ctx != nullptr && ctx->device != nullptr &&
        ctx->device->graphicsQueue() != ctx->device->transferQueue());

    LinearAllocator scratchBacking{sLoadingScratchSize};
    ScopedScratch scopeAlloc{scratchBacking};

    ctx->meshTimer.reset();
    while (!ctx->interruptLoading)
    {
        if (ctx->workerLoadedMeshCount < ctx->meshes.size())
            loadNextMesh(scopeAlloc.child_scope(), ctx);
        else
            loadNextTexture(scopeAlloc.child_scope(), ctx);

        // TODO:
        // Can we just store the high watermark? The allocator doesn't get
        // recreated between loops
        const uint32_t previousHighWatermark =
            ctx->linearAllocatorHighWatermark;
        if (scratchBacking.allocated_byte_count_high_watermark() >
            previousHighWatermark)
        {
            ctx->linearAllocatorHighWatermark = asserted_cast<uint32_t>(
                scratchBacking.allocated_byte_count_high_watermark());
        }
        ctx->generalAllocatorHightWatermark = asserted_cast<uint32_t>(
            ctx->alloc.stats().allocated_byte_count_high_watermark);
    }
}

} // namespace

Buffer createTextureStaging(Device *device)
{
    // Assume at most 4k at 8bits per channel
    const vk::DeviceSize stagingSize = static_cast<size_t>(4096) *
                                       static_cast<size_t>(4096) *
                                       sizeof(uint32_t);
    return device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = stagingSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = "Texture2DStaging",
    });
}

DeferredLoadingContext::DeferredLoadingContext(
    Device *device, std::filesystem::path sceneDir,
    std::filesystem::file_time_type sceneWriteTime,
    const tinygltf::Model &gltfModel)
: device{device}
, sceneDir{WHEELS_MOV(sceneDir)}
, sceneWriteTime{sceneWriteTime}
, alloc{megabytes(1)}
, gltfModel{gltfModel}
, cb{device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
      .commandPool = device->transferPool(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1})[0]}
, meshes{alloc}
, loadedMeshes{alloc, gltfModel.meshes.size()}
, loadedTextures{alloc, gltfModel.images.size()}
, materials{alloc, gltfModel.materials.size()}
{
    WHEELS_ASSERT(device != nullptr);

    // One of these is used by the worker implementation, all by the
    // single threaded one
    for (uint32_t i = 0; i < stagingBuffers.capacity(); ++i)
        stagingBuffers[i] = createTextureStaging(device);

    geometryUploadBuffer = device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sGeometryBufferSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .createMapped = true,
        .debugName = "GeometryUploadBuffer",
    });
}

DeferredLoadingContext::~DeferredLoadingContext()
{
    if (device != nullptr)
    {
        if (worker.has_value())
        {
            interruptLoading = true;
            worker->join();
        }

        for (const Buffer &buffer : stagingBuffers)
            device->destroy(buffer);

        device->destroy(geometryUploadBuffer);
    }
}

void DeferredLoadingContext::launch()
{
    WHEELS_ASSERT(
        !worker.has_value() && "Tried to launch deferred loading worker twice");
    worker = std::thread{&loadingWorker, this};
}

UploadedGeometryData DeferredLoadingContext::uploadGeometryData(
    const MeshCacheHeader &cacheHeader, const Array<uint8_t> &dataBlob)
{
    WHEELS_ASSERT(cacheHeader.blobByteCount > 0);
    WHEELS_ASSERT(cacheHeader.blobByteCount == dataBlob.size());

    const uint32_t dstBufferI = getGeometryBuffer(cacheHeader.blobByteCount);

    // The mesh data ranges are expected to not leave gaps in the buffer so that
    // ownership is transferred properly between the queues.
    const uint32_t startByteOffset =
        (sGeometryBufferSize - geometryBufferRemainingByteCounts[dstBufferI]);
    WHEELS_ASSERT(
        startByteOffset % sizeof(uint32_t) == 0 &&
        "Mesh data should be aligned for u32");

    uint32_t *dstPtr =
        reinterpret_cast<uint32_t *>(geometryUploadBuffer.mapped);
    memcpy(dstPtr, dataBlob.data(), cacheHeader.blobByteCount);

    const vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = startByteOffset,
        .size = cacheHeader.blobByteCount,
    };
    const Buffer &dstBuffer = geometryBuffers[dstBufferI];
    // Don't use the context command buffer since this method is also used
    // by the non-async loading implementations
    cb.copyBuffer(
        geometryUploadBuffer.handle, dstBuffer.handle, 1, &copyRegion);

    // The mesh data ranges are expected to not leave gaps in the buffer so
    // that ownership is transferred properly between the queues. Any
    // alignment/padding for the next mesh should be included in the byte
    // count of the previous one.
    geometryBufferRemainingByteCounts[dstBufferI] -= cacheHeader.blobByteCount;

    // Offsets into GPU buffer are for u32
    const uint32_t startOffsetU32 =
        startByteOffset / asserted_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t startOffsetU16 =
        startByteOffset / asserted_cast<uint32_t>(sizeof(uint16_t));

    const UploadedGeometryData ret{
        .metadata =
            GeometryMetadata{
                .bufferIndex = dstBufferI,
                .indicesOffset =
                    (cacheHeader.usesShortIndices == 1 ? startOffsetU16
                                                       : startOffsetU32),
                .positionsOffset = startOffsetU32 + cacheHeader.positionsOffset,
                .normalsOffset = startOffsetU32 + cacheHeader.normalsOffset,
                .tangentsOffset =
                    cacheHeader.tangentsOffset == 0xFFFFFFFF
                        ? 0xFFFFFFFF
                        : startOffsetU32 + cacheHeader.tangentsOffset,
                .texCoord0sOffset =
                    cacheHeader.texCoord0sOffset == 0xFFFFFFFF
                        ? 0xFFFFFFFF
                        : startOffsetU32 + cacheHeader.texCoord0sOffset,
                .usesShortIndices = cacheHeader.usesShortIndices,
            },
        .byteCount = cacheHeader.blobByteCount,
    };

    return ret;
}

uint32_t DeferredLoadingContext::getGeometryBuffer(uint32_t byteCount)
{
    // Find a buffer that fits the data or create a new one
    // Let's assume there's only a handful of these so we can just comb
    // through all of them and potentially fill early buffers more
    // completely than if we just checked the last one.
    uint32_t dstBufferI = 0;
    WHEELS_ASSERT(
        geometryBuffers.size() == geometryBufferRemainingByteCounts.size());
    const uint32_t bufferCount =
        asserted_cast<uint32_t>(geometryBufferRemainingByteCounts.size());
    for (; dstBufferI < bufferCount; ++dstBufferI)
    {
        const uint32_t bc = geometryBufferRemainingByteCounts[dstBufferI];
        if (bc >= byteCount)
            break;
    }

    WHEELS_ASSERT(byteCount <= sGeometryBufferSize);

    if (dstBufferI >= geometryBuffers.size())
    {
        Buffer buffer = device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = sGeometryBufferSize,
                    .usage = vk::BufferUsageFlagBits::
                                 eAccelerationStructureBuildInputReadOnlyKHR |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .cacheDeviceAddress = true,
            .debugName = "GeometryBuffer",
        });
        {
            // The managing thread should only read the buffer array. A lock
            // is only be needed for the append op on the worker side to
            // sync those reads.
            const std::lock_guard _lock{geometryBuffersMutex};
            geometryBuffers.push_back(WHEELS_MOV(buffer));
        }
        geometryBufferRemainingByteCounts.push_back(sGeometryBufferSize);
    }

    return dstBufferI;
}
