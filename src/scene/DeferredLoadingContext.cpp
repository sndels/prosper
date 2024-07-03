#include "DeferredLoadingContext.hpp"

#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include "../utils/Logger.hpp"
#include "../utils/Utils.hpp"
#include <glm/gtc/packing.hpp>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <wheels/allocators/linear_allocator.hpp>

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sGeometryBufferSize = asserted_cast<uint32_t>(megabytes(64));

const uint64_t sMeshCacheMagic = 0x4853'4D52'5053'5250; // PRSPRMSH
// This should be incremented when breaking changes are made to
// what's cached
const uint32_t sMeshCacheVersion = 4;

// Balance between cluster size and cone culling efficiency
const float sConeWeight = 0.5f;

// Need to pass the allocator with function pointers that don't have userdata
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Allocator *sMeshoptAllocator = nullptr;

template <typename T>
void remapVertexAttribute(
    Array<T> &src, const Array<uint32_t> &remapIndices,
    size_t uniqueVertexCount)
{
    Array<T> remapped{gAllocators.loadingWorker};
    remapped.resize(uniqueVertexCount);
    meshopt_remapVertexBuffer(
        remapped.data(), src.data(), src.size(), sizeof(T),
        remapIndices.data());

    src = WHEELS_MOV(remapped);
}

struct MeshletBounds
{
    // Bounding sphere
    vec3 center{};
    float radius{0.f};
    // Normal cone
    i8vec3 coneAxisS8{};
    int8_t coneCutoffS8{0};
};
static_assert(sizeof(MeshletBounds) == 5 * sizeof(uint32_t));
static_assert(offsetof(MeshletBounds, coneAxisS8) == 4 * sizeof(uint32_t));
static_assert(
    offsetof(MeshletBounds, coneCutoffS8) ==
    4 * sizeof(uint32_t) + 3 * sizeof(int8_t));

static_assert(
    sizeof(meshopt_Meshlet) == 4 * sizeof(uint32_t),
    "Mesh shaders use meshoptimizer meshlets as is.");

struct MeshData
{
    wheels::Array<uint32_t> indices;
    wheels::Array<glm::vec3> positions;
    wheels::Array<glm::vec3> normals;
    wheels::Array<glm::vec4> tangents;
    wheels::Array<glm::vec2> texCoord0s;
    wheels::Array<meshopt_Meshlet> meshlets;
    wheels::Array<MeshletBounds> meshletBounds;
    wheels::Array<uint32_t> meshletVertices;
    wheels::Array<uint8_t> meshletTriangles;
};

struct PackedMeshData
{
    wheels::Array<uint32_t> indices{gAllocators.loadingWorker};
    // Packed as r16g16b16a16_sfloat
    // TODO:
    // Pack as r16g16b16a16_snorm relative to object space AABB to have uniform
    // (and potentially better) precision. Unpacking would then be pos *
    // aabbHalfAxisOS + aabbCenterOS and it can be concatenated into the
    // objectToWorld transform (careful to not include it in parent transforms)
    wheels::Array<uint64_t> positions{gAllocators.loadingWorker};
    // Packed as r10g10b10(a2)_snorm
    wheels::Array<uint32_t> normals{gAllocators.loadingWorker};
    // Packed as r10g10b10a2_snorm, sign in a2
    wheels::Array<uint32_t> tangents{gAllocators.loadingWorker};
    // Packed as r16g16_sfloat
    wheels::Array<uint32_t> texCoord0s{gAllocators.loadingWorker};
    wheels::Array<meshopt_Meshlet> meshlets{gAllocators.loadingWorker};
    wheels::Array<MeshletBounds> meshletBounds{gAllocators.loadingWorker};
    wheels::Array<uint32_t> meshletVertices{gAllocators.loadingWorker};
    wheels::Array<uint8_t> meshletTriangles{gAllocators.loadingWorker};
};
static_assert(
    sizeof(meshopt_Meshlet) == 4 * sizeof(uint32_t),
    "Mesh shaders use meshoptimizer meshlets as is.");
static_assert(sizeof(MeshletBounds) == 5 * sizeof(uint32_t));
static_assert(offsetof(MeshletBounds, coneAxisS8) == 4 * sizeof(uint32_t));
static_assert(
    offsetof(MeshletBounds, coneCutoffS8) ==
    4 * sizeof(uint32_t) + 3 * sizeof(int8_t));

template <int N>
void unpackVector(
    const cgltf_accessor *accessor, Array<vec<N, float, defaultp>> &out)
{
    WHEELS_ASSERT(accessor != nullptr);
    out.resize(accessor->count);
    const cgltf_size unpackedCount = cgltf_accessor_unpack_floats(
        accessor, &out.data()[0][0], out.size() * N);
    WHEELS_ASSERT(unpackedCount == out.size() * N);
}

MeshData getMeshData(
    const InputGeometryMetadata &metadata, const MeshInfo &meshInfo)
{
    MeshData ret{
        .indices = Array<uint32_t>{gAllocators.loadingWorker},
        .positions = Array<vec3>{gAllocators.loadingWorker},
        .normals = Array<vec3>{gAllocators.loadingWorker},
        .tangents = Array<vec4>{gAllocators.loadingWorker},
        .texCoord0s = Array<vec2>{gAllocators.loadingWorker},
        .meshlets = Array<meshopt_Meshlet>{gAllocators.loadingWorker},
        .meshletBounds = Array<MeshletBounds>{gAllocators.loadingWorker},
        .meshletVertices = Array<uint32_t>{gAllocators.loadingWorker},
        .meshletTriangles = Array<uint8_t>{gAllocators.loadingWorker},
    };

    {
        WHEELS_ASSERT(metadata.indices != nullptr);
        WHEELS_ASSERT(meshInfo.indexCount == metadata.indices->count);
        ret.indices.resize(meshInfo.indexCount);
        const cgltf_size unpackedCount = cgltf_accessor_unpack_indices(
            metadata.indices, ret.indices.data(), sizeof(ret.indices[0]),
            ret.indices.size());
        WHEELS_ASSERT(unpackedCount == meshInfo.indexCount);
    }

    unpackVector(metadata.positions, ret.positions);
    unpackVector(metadata.normals, ret.normals);
    if (metadata.tangents != nullptr)
        unpackVector(metadata.tangents, ret.tangents);
    if (metadata.texCoord0s != nullptr)
        unpackVector(metadata.texCoord0s, ret.texCoord0s);

    return ret;
}

// mikktspace defined interface
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,bugprone-easily-swappable-parameters)

int mikkTGetNumFaces(const SMikkTSpaceContext *pContext)
{
    WHEELS_ASSERT(pContext != nullptr);
    WHEELS_ASSERT(pContext->m_pUserData != nullptr);

    const MeshData *meshData =
        static_cast<const MeshData *>(pContext->m_pUserData);

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
        static_cast<const MeshData *>(pContext->m_pUserData);

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
        static_cast<const MeshData *>(pContext->m_pUserData);

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
        static_cast<const MeshData *>(pContext->m_pUserData);

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

    MeshData *meshData = static_cast<MeshData *>(pContext->m_pUserData);

    const int vertexI = mikkTVertexIndex(iFace, iVert);
    meshData->tangents[vertexI] =
        vec4{fvTangent[0], fvTangent[1], fvTangent[2], fSign};
}

// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,bugprone-easily-swappable-parameters)

template <typename T>
void flattenAttribute(Array<T> &attribute, const Array<uint32_t> &indices)
{
    Array<T> flattened{gAllocators.loadingWorker, indices.size()};
    for (const uint32_t i : indices)
        flattened.push_back(attribute[i]);

    attribute = WHEELS_MOV(flattened);
}

void generateTangents(MeshData *meshData)
{
    WHEELS_ASSERT(meshData != nullptr);
    WHEELS_ASSERT(meshData->tangents.empty());
    WHEELS_ASSERT(meshData->positions.size() == meshData->normals.size());
    WHEELS_ASSERT(meshData->positions.size() == meshData->texCoord0s.size());

    // Flatten data first as instructed in the mikktspace header
    // TODO: tmp buffers here
    const size_t flattenedVertexCount = meshData->indices.size();
    flattenAttribute(meshData->positions, meshData->indices);
    flattenAttribute(meshData->normals, meshData->indices);
    flattenAttribute(meshData->texCoord0s, meshData->indices);
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

    Array<uint32_t> remapTable{gAllocators.loadingWorker};
    remapTable.resize(flattenedVertexCount);
    const size_t uniqueVertexCount = meshopt_generateVertexRemapMulti(
        remapTable.data(), nullptr, flattenedVertexCount, flattenedVertexCount,
        vertexStreams.data(), vertexStreams.size());

    meshData->indices.resize(flattenedVertexCount);
    meshopt_remapIndexBuffer(
        meshData->indices.data(), nullptr, flattenedVertexCount,
        remapTable.data());

    remapVertexAttribute(meshData->positions, remapTable, uniqueVertexCount);
    remapVertexAttribute(meshData->normals, remapTable, uniqueVertexCount);
    remapVertexAttribute(meshData->tangents, remapTable, uniqueVertexCount);
    remapVertexAttribute(meshData->texCoord0s, remapTable, uniqueVertexCount);
}

void optimizeMeshData(
    MeshData *meshData, MeshInfo *meshInfo, const char *meshName)
{
    WHEELS_ASSERT(meshData != nullptr);

    const size_t indexCount = meshData->indices.size();
    const size_t vertexCount = meshData->positions.size();

    Array<uint32_t> tmpIndices{gAllocators.loadingWorker};
    tmpIndices.resize(meshData->indices.size());
    meshopt_optimizeVertexCache(
        tmpIndices.data(), meshData->indices.data(), indexCount, vertexCount);

    const float vertexCacheDegradationThreshod = 1.00f;
    meshopt_optimizeOverdraw(
        meshData->indices.data(), tmpIndices.data(), tmpIndices.size(),
        &meshData->positions.data()[0].x, vertexCount, sizeof(vec3),
        vertexCacheDegradationThreshod);

    Array<uint32_t> remapIndices{gAllocators.loadingWorker};
    remapIndices.resize(vertexCount);
    const size_t uniqueVertexCount = meshopt_optimizeVertexFetchRemap(
        remapIndices.data(), meshData->indices.data(), indexCount, vertexCount);
    if (uniqueVertexCount < vertexCount)
        LOG_WARN("Mesh '%s' has unused vertices", meshName);

    // Reuse tmpIndices as it's not required after optimizeOverdraw
    meshopt_remapIndexBuffer(
        tmpIndices.data(), meshData->indices.data(), indexCount,
        remapIndices.data());
    meshData->indices = WHEELS_MOV(tmpIndices);

    remapVertexAttribute(meshData->positions, remapIndices, uniqueVertexCount);
    remapVertexAttribute(meshData->normals, remapIndices, uniqueVertexCount);
    remapVertexAttribute(meshData->tangents, remapIndices, uniqueVertexCount);
    remapVertexAttribute(meshData->texCoord0s, remapIndices, uniqueVertexCount);

    meshInfo->vertexCount = asserted_cast<uint32_t>(uniqueVertexCount);
}

void generateMeshlets(MeshData *meshData)
{
    WHEELS_ASSERT(meshData != nullptr);
    WHEELS_ASSERT(meshData->meshlets.empty());
    WHEELS_ASSERT(meshData->meshletVertices.empty());
    WHEELS_ASSERT(meshData->meshletTriangles.empty());

    const size_t maxMeshlets = meshopt_buildMeshletsBound(
        meshData->indices.size(), sMaxMsVertices, sMaxMsTriangles);
    WHEELS_ASSERT(maxMeshlets > 0);

    meshData->meshlets.resize(maxMeshlets);
    meshData->meshletVertices.resize(maxMeshlets * sMaxMsVertices);
    meshData->meshletTriangles.resize(maxMeshlets * sMaxMsTriangles * 3);

    const size_t meshletCount = meshopt_buildMeshlets(
        meshData->meshlets.data(), meshData->meshletVertices.data(),
        meshData->meshletTriangles.data(), meshData->indices.data(),
        meshData->indices.size(), &meshData->positions[0].x,
        meshData->positions.size(), sizeof(vec3), sMaxMsVertices,
        sMaxMsTriangles, sConeWeight);
    WHEELS_ASSERT(meshletCount > 0);

    // Need to trim the buffers now that we know the tight sizes
    meshData->meshlets.resize(meshletCount);

    const meshopt_Meshlet &lastMeshlet = meshData->meshlets.back();

    meshData->meshletVertices.resize(
        lastMeshlet.vertex_offset + lastMeshlet.vertex_count);
    // Pad up to a u32 boundary
    const uint32_t trianglesSize =
        asserted_cast<uint32_t>(wheels::aligned_offset(
            lastMeshlet.triangle_offset + lastMeshlet.triangle_count,
            sizeof(uint32_t)));
    meshData->meshletTriangles.resize(asserted_cast<size_t>(trianglesSize) * 3);

    meshData->meshletBounds.reserve(meshData->meshlets.size());
    for (const meshopt_Meshlet &meshlet : meshData->meshlets)
    {
        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            &meshData->meshletVertices[meshlet.vertex_offset],
            &meshData->meshletTriangles[meshlet.triangle_offset],
            meshlet.triangle_count, &meshData->positions[0].x,
            meshData->positions.size(), sizeof(vec3));
        meshData->meshletBounds.push_back(MeshletBounds{
            .center =
                vec3{
                    bounds.center[0],
                    bounds.center[1],
                    bounds.center[2],
                },
            .radius = bounds.radius,
            .coneAxisS8 =
                i8vec3{
                    bounds.cone_axis_s8[0],
                    bounds.cone_axis_s8[1],
                    bounds.cone_axis_s8[2],
                },
            .coneCutoffS8 = bounds.cone_cutoff_s8,
        });
    }
}

PackedMeshData packMeshData(MeshData &&meshData)
{
    PackedMeshData ret;

    ret.indices = WHEELS_MOV(meshData.indices);

    ret.positions.reserve(meshData.positions.size());
    for (const vec3 &p : meshData.positions)
    {
        static_assert(
            sVertexPositionFormat == vk::Format::eR16G16B16A16Sfloat,
            "Packing doesn't match the global format");
        ret.positions.push_back(packHalf4x16(vec4{p, 1.f}));
    }

    ret.normals.reserve(meshData.normals.size());
    for (const vec3 &n : meshData.normals)
    {
        static_assert(
            sVertexNormalFormat == vk::Format::eA2B10G10R10SnormPack32,
            "Packing doesn't match the global format");
        ret.normals.push_back(packSnorm3x10_1x2(vec4{n, 0.f}));
    }

    ret.tangents.reserve(meshData.tangents.size());
    for (const vec4 &ts : meshData.tangents)
    {
        static_assert(
            sVertexTangentFormat == vk::Format::eA2B10G10R10SnormPack32,
            "Packing doesn't match the global format");
        ret.tangents.push_back(packSnorm3x10_1x2(ts));
    }

    ret.texCoord0s.reserve(meshData.texCoord0s.size());
    for (const vec2 &uv : meshData.texCoord0s)
    {
        static_assert(
            sVertexTexCoord0Format == vk::Format::eR16G16Sfloat,
            "Packing doesn't match the global format");
        ret.texCoord0s.push_back(packHalf2x16(uv));
    }

    ret.meshlets = WHEELS_MOV(meshData.meshlets);
    ret.meshletBounds = WHEELS_MOV(meshData.meshletBounds);
    ret.meshletVertices = WHEELS_MOV(meshData.meshletVertices);
    ret.meshletTriangles = WHEELS_MOV(meshData.meshletTriangles);

    return ret;
}

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
        LOG_INFO("Missing cache for %s", cachePath.string().c_str());
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
        LOG_INFO("Old cache data version for %s", cachePath.string().c_str());
        return ret;
    }

    ret = MeshCacheHeader{};

    readRaw(cacheFile, ret->sourceWriteTime);
    readRaw(cacheFile, ret->indexCount);
    readRaw(cacheFile, ret->vertexCount);
    readRaw(cacheFile, ret->meshletCount);
    readRaw(cacheFile, ret->positionsOffset);
    readRaw(cacheFile, ret->normalsOffset);
    readRaw(cacheFile, ret->tangentsOffset);
    readRaw(cacheFile, ret->texCoord0sOffset);
    readRaw(cacheFile, ret->meshletsOffset);
    readRaw(cacheFile, ret->meshletBoundsOffset);
    readRaw(cacheFile, ret->meshletVerticesOffset);
    readRaw(cacheFile, ret->meshletTrianglesByteOffset);
    readRaw(cacheFile, ret->usesShortIndices);
    readRaw(cacheFile, ret->blobByteCount);

    if (dataBlobOut != nullptr)
    {
        dataBlobOut->resize(ret->blobByteCount);
        readRawSpan(cacheFile, dataBlobOut->mut_span());
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
        LOG_INFO("Stale cache for %s", cachePath.string().c_str());
        return false;
    }
    return true;
}

Optional<uint32_t> getImageIndex(
    const cgltf_data *gltfData, const cgltf_texture *texture)
{
    if (texture == nullptr || texture->image == nullptr)
        return {};

    return asserted_cast<uint32_t>(cgltf_image_index(gltfData, texture->image));
}

void printImageColorSpaceReuseWarning(const cgltf_image *image)
{
    const char *debugName = nullptr;
    if (image != nullptr)
    {
        if (image->uri != nullptr)
            debugName = image->uri;
        else if (image->name != nullptr)
            debugName = image->name;
    }
    if (debugName != nullptr)
        LOG_WARN(
            "'%s' is used both as a linear and sRgb texture. Mip maps will be "
            "generated with sRgb filtering",
            debugName);
    else
        // We shouldn't really get here with decent files, but let's still log
        // that there is an issue
        LOG_WARN("An unnamed image is used both as a linear and sRgb texture. "
                 "Mip maps will be generated with sRgb filtering");
}

void writeCache(
    const std::filesystem::path &sceneDir,
    std::filesystem::file_time_type sceneWriteTime, uint32_t meshIndex,
    PackedMeshData &&meshData, const MeshInfo &meshInfo)
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
    Array<uint8_t> packedIndices{gAllocators.loadingWorker};
    Array<uint8_t> packedMeshletVertices{gAllocators.loadingWorker};
    if (usesShortIndices)
    {
        {
            size_t byteCount = meshInfo.indexCount * sizeof(uint16_t);
            // Let's pad to 4byte boundary to make things simpler later
            byteCount = aligned_offset(byteCount, sizeof(uint32_t));
            WHEELS_ASSERT(byteCount % sizeof(uint32_t) == 0);

            packedIndices.resize(byteCount);
            uint16_t *dst = reinterpret_cast<uint16_t *>(packedIndices.data());
            for (size_t i = 0; i < meshInfo.indexCount; ++i)
                dst[i] = asserted_cast<uint16_t>(meshData.indices[i]);
        }
        {
            size_t byteCount =
                meshData.meshletVertices.size() * sizeof(uint16_t);
            // Let's pad to 4byte boundary to make things simpler later
            byteCount = aligned_offset(byteCount, sizeof(uint32_t));
            WHEELS_ASSERT(byteCount % sizeof(uint32_t) == 0);

            packedMeshletVertices.resize(byteCount);
            uint16_t *dst =
                reinterpret_cast<uint16_t *>(packedMeshletVertices.data());
            const size_t count = meshData.meshletVertices.size();
            for (size_t i = 0; i < count; ++i)
                dst[i] = asserted_cast<uint16_t>(meshData.meshletVertices[i]);
            // Original offsets are ok as we'll read this as u16 in shader
        }
    }
    else
    {
        // TODO:
        // Just use the data directly instead the extra allocation and
        // memcpy?
        {
            const size_t byteCount = meshInfo.indexCount * sizeof(uint32_t);
            packedIndices.resize(byteCount);
            memcpy(packedIndices.data(), meshData.indices.data(), byteCount);
        }
        {
            const size_t byteCount = meshInfo.indexCount * sizeof(uint32_t);
            packedMeshletVertices.resize(byteCount);
            memcpy(
                packedMeshletVertices.data(), meshData.meshletVertices.data(),
                byteCount);
        }
    }

    uint32_t byteCount = asserted_cast<uint32_t>(
        packedIndices.size() * sizeof(decltype(packedIndices)::value_type));

    auto computeOffset =
        [&byteCount]<typename T>(const Array<T> &data) -> uint32_t
    {
        const uint32_t offset = byteCount;
        byteCount += asserted_cast<uint32_t>(data.size() * sizeof(T));
        WHEELS_ASSERT(
            byteCount % sizeof(uint32_t) == 0 &&
            "Mesh data is not aligned properly");
        return offset;
    };

    // Many offsets into the GPU buffer are for u32
    const uint32_t elementSize = static_cast<uint32_t>(sizeof(uint32_t));

    // Figure out the offsets and total byte count
    // NOTE: Order here has to match the write order into the file
    const uint32_t positionsOffset =
        computeOffset(meshData.positions) / elementSize;
    const uint32_t normalsOffset =
        computeOffset(meshData.normals) / elementSize;
    const uint32_t tangentsOffset =
        computeOffset(meshData.tangents) / elementSize;
    const uint32_t texCoord0sOffset =
        computeOffset(meshData.texCoord0s) / elementSize;
    const uint32_t meshletsOffset =
        computeOffset(meshData.meshlets) / elementSize;
    const uint32_t meshletBoundsOffset =
        computeOffset(meshData.meshletBounds) / elementSize;
    const uint32_t meshletVerticesOffset =
        computeOffset(packedMeshletVertices) /
        static_cast<uint32_t>(
            usesShortIndices ? sizeof(uint16_t) : sizeof(uint32_t));
    const uint32_t meshletTrianglesOffset =
        computeOffset(meshData.meshletTriangles);

    WHEELS_ASSERT(
        byteCount % sizeof(uint32_t) == 0 &&
        "Mesh data is not aligned properly");
    WHEELS_ASSERT(
        byteCount < sGeometryBufferSize &&
        "The default size for geometry buffers doesn't fit the mesh");

    const MeshCacheHeader header{
        .sourceWriteTime = sceneWriteTime,
        .indexCount = meshInfo.indexCount,
        .vertexCount = meshInfo.vertexCount,
        .meshletCount = asserted_cast<uint32_t>(meshData.meshlets.size()),
        .positionsOffset = positionsOffset,
        .normalsOffset = normalsOffset,
        .tangentsOffset = hasTangents ? tangentsOffset : 0xFFFF'FFFF,
        .texCoord0sOffset = hasTexCoord0s ? texCoord0sOffset : 0xFFFF'FFFF,
        .meshletsOffset = meshletsOffset,
        .meshletBoundsOffset = meshletBoundsOffset,
        .meshletVerticesOffset = meshletVerticesOffset,
        .meshletTrianglesByteOffset = meshletTrianglesOffset,
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
    writeRaw(cacheFile, header.meshletCount);
    writeRaw(cacheFile, header.positionsOffset);
    writeRaw(cacheFile, header.normalsOffset);
    writeRaw(cacheFile, header.tangentsOffset);
    writeRaw(cacheFile, header.texCoord0sOffset);
    writeRaw(cacheFile, header.meshletsOffset);
    writeRaw(cacheFile, header.meshletBoundsOffset);
    writeRaw(cacheFile, header.meshletVerticesOffset);
    writeRaw(cacheFile, header.meshletTrianglesByteOffset);
    writeRaw(cacheFile, header.usesShortIndices);
    writeRaw(cacheFile, header.blobByteCount);

    const std::streampos blobStart = cacheFile.tellp();
    writeRawSpan(cacheFile, packedIndices.span());
    writeRawSpan(cacheFile, meshData.positions.span());
    writeRawSpan(cacheFile, meshData.normals.span());
    writeRawSpan(cacheFile, meshData.tangents.span());
    writeRawSpan(cacheFile, meshData.texCoord0s.span());
    writeRawSpan(cacheFile, meshData.meshlets.span());
    writeRawSpan(cacheFile, meshData.meshletBounds.span());
    writeRawSpan(cacheFile, packedMeshletVertices.span());
    writeRawSpan(cacheFile, meshData.meshletTriangles.span());
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

void loadNextMesh(DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);

    // Set up a custom allocator for meshopt, let's keep track of allocations
    // there too
    sMeshoptAllocator = &gAllocators.loadingWorker;
    auto meshoptAllocate = [](size_t byteCount) -> void *
    { return sMeshoptAllocator->allocate(byteCount); };
    auto meshoptDeallocate = [](void *ptr)
    { sMeshoptAllocator->deallocate(ptr); };
    meshopt_setAllocator(meshoptAllocate, meshoptDeallocate);

    const uint32_t meshIndex = ctx->workerLoadedMeshCount;
    WHEELS_ASSERT(meshIndex < ctx->meshes.size());

    // Ctx member functions will use the command buffer
    ctx->cb.reset();
    ctx->cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    const QueueFamilies &families = gDevice.queueFamilies();
    WHEELS_ASSERT(families.graphicsFamily.has_value());
    WHEELS_ASSERT(families.transferFamily.has_value());

    const Pair<InputGeometryMetadata, MeshInfo> &nextMesh =
        ctx->meshes[meshIndex];
    const InputGeometryMetadata &metadata = nextMesh.first;
    MeshInfo info = nextMesh.second;

    const char *meshName = ctx->gltfData->meshes[metadata.sourceMeshIndex].name;
    ctx->meshNames.emplace_back(gAllocators.loadingWorker, meshName);

    const std::filesystem::path cachePath =
        getCachePath(ctx->sceneDir, meshIndex);
    if (!cacheValid(cachePath, ctx->sceneWriteTime))
    {
        MeshData meshData = getMeshData(metadata, info);

        if (meshData.tangents.empty() && !meshData.texCoord0s.empty())
        {
            generateTangents(&meshData);
            info.vertexCount =
                asserted_cast<uint32_t>(meshData.positions.size());
        }

        optimizeMeshData(&meshData, &info, meshName);

        generateMeshlets(&meshData);

        PackedMeshData packedMeshData = packMeshData(WHEELS_MOV(meshData));

        writeCache(
            ctx->sceneDir, ctx->sceneWriteTime, meshIndex,
            WHEELS_MOV(packedMeshData), info);
    }

    // Always read from the cache to make caching issues always visible
    Array<uint8_t> dataBlob{gAllocators.loadingWorker};
    const Optional<MeshCacheHeader> cacheHeader =
        readCache(cachePath, &dataBlob);
    WHEELS_ASSERT(cacheHeader.has_value());
    WHEELS_ASSERT(cacheHeader->indexCount == info.indexCount);
    // Tangent generation can change vertex count
    info.vertexCount = cacheHeader->vertexCount;
    info.meshletCount = cacheHeader->meshletCount;

    const UploadedGeometryData uploadData = ctx->uploadGeometryData(
        *cacheHeader, dataBlob, ctx->meshNames[meshIndex]);

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
            .offset = uploadData.byteOffset,
            .size = uploadData.byteCount,
        };
        ctx->cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &releaseBarrier,
        });
    }

    ctx->cb.end();

    const vk::Queue transferQueue = gDevice.transferQueue();
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
        const std::lock_guard lock{ctx->loadedMeshesMutex};

        ctx->loadedMeshes.emplace_back(uploadData, info);
    }

    if (ctx->workerLoadedMeshCount == ctx->meshes.size())
    {
        LOG_INFO("Mesh loading took %.2fs", ctx->meshTimer.getSeconds());
        ctx->textureTimer.reset();
    }
}

void loadNextTexture(DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);

    const uint32_t imageIndex = ctx->workerLoadedImageCount;
    if (imageIndex == ctx->gltfData->images_count)
    {
        LOG_INFO("Texture loading took %.2fs", ctx->textureTimer.getSeconds());
        ctx->interruptLoading = true;
        return;
    }

    WHEELS_ASSERT(ctx->gltfData->images_count > imageIndex);
    const cgltf_image &image = ctx->gltfData->images[imageIndex];
    if (image.uri == nullptr)
        throw std::runtime_error("Embedded glTF textures aren't supported. "
                                 "Scene should be glTF + bin + textures.");

    ctx->cb.reset();
    ctx->cb.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    LinearAllocator scopeBacking{
        gAllocators.loadingWorker, Allocators::sLoadingScratchSize};

    TextureColorSpace colorSpace = TextureColorSpace::sRgb;
    if (ctx->linearColorImages.contains(imageIndex))
    {
        WHEELS_ASSERT(
            !ctx->sRgbColorImages.contains(imageIndex) &&
            "Image should belong to exactly one colorspace set");
        colorSpace = TextureColorSpace::Linear;
    }
    else
        WHEELS_ASSERT(
            ctx->sRgbColorImages.contains(imageIndex) &&
            "Image should belong to exactly one colorspace set");

    Texture2D tex;
    tex.init(
        ScopedScratch{scopeBacking}, ctx->sceneDir / image.uri, ctx->cb,
        ctx->stagingBuffers[0],
        Texture2DOptions{
            .generateMipMaps = true,
            .colorSpace = colorSpace,
        });

    const QueueFamilies &families = gDevice.queueFamilies();
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

    const vk::Queue transferQueue = gDevice.transferQueue();
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
        const std::lock_guard lock{ctx->loadedTexturesMutex};

        ctx->loadedTextures.emplace_back(WHEELS_MOV(tex));
    }
}

void loadingWorker(DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);
    WHEELS_ASSERT(gDevice.graphicsQueue() != gDevice.transferQueue());

    setCurrentThreadName("prosper loading");

    ctx->meshTimer.reset();
    while (!ctx->interruptLoading)
    {
        if (ctx->workerLoadedMeshCount < ctx->meshes.size())
        {
            loadNextMesh(ctx);

            // Only update for meshes as textures will always allocate a big
            // worst case tmp chunk for linear allocation
            gAllocators.loadingWorkerHighWatermark =
                gAllocators.loadingWorker.stats()
                    .allocated_byte_count_high_watermark;
        }
        else
            loadNextTexture(ctx);
    }
}

} // namespace

Buffer createTextureStaging()
{
    // Assume at most 4k at 8bits per channel
    const vk::DeviceSize stagingSize = static_cast<size_t>(4096) *
                                       static_cast<size_t>(4096) *
                                       sizeof(uint32_t);
    return gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = stagingSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "Texture2DStaging",
    });
}

DeferredLoadingContext::~DeferredLoadingContext()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    if (worker.has_value())
    {
        interruptLoading = true;
        worker->join();
    }

    for (Buffer &buffer : stagingBuffers)
        gDevice.destroy(buffer);

    gDevice.destroy(geometryUploadBuffer);
    cgltf_free(gltfData);
}

void DeferredLoadingContext::init(
    std::filesystem::path inSceneDir,
    std::filesystem::file_time_type inSceneWriteTime, cgltf_data *inGltfData)
{
    WHEELS_ASSERT(!initialized);
    WHEELS_ASSERT(inGltfData != nullptr);

    sceneDir = WHEELS_MOV(inSceneDir);
    sceneWriteTime = inSceneWriteTime;
    gltfData = inGltfData;
    cb = gDevice.logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool = gDevice.transferPool(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1})[0];
    loadedMeshes.reserve(gltfData->meshes_count);
    loadedTextures.reserve(gltfData->images_count);
    materials.reserve(gltfData->materials_count);

    // One of these is used by the worker implementation, all by the
    // single threaded one
    for (uint32_t i = 0; i < stagingBuffers.capacity(); ++i)
        stagingBuffers[i] = createTextureStaging();

    geometryUploadBuffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sGeometryBufferSize,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "GeometryUploadBuffer",
    });

    initialized = true;
}

void DeferredLoadingContext::launch()
{
    WHEELS_ASSERT(initialized);
    WHEELS_ASSERT(
        !worker.has_value() && "Tried to launch deferred loading worker twice");

    // Fill sets to query image colorspaces from
    for (const cgltf_material &material :
         Span{gltfData->materials, gltfData->materials_count})
    {
        if (material.has_pbr_metallic_roughness == 1)
        {
            const cgltf_pbr_metallic_roughness &pbrParams =
                material.pbr_metallic_roughness;

            const Optional<uint32_t> baseColorIndex =
                getImageIndex(gltfData, pbrParams.base_color_texture.texture);
            if (baseColorIndex.has_value())
            {
                if (linearColorImages.contains(*baseColorIndex))
                {
                    printImageColorSpaceReuseWarning(
                        gltfData->textures[*baseColorIndex].image);
                    linearColorImages.remove(*baseColorIndex);
                }
                sRgbColorImages.insert(*baseColorIndex);
            }

            const Optional<uint32_t> metallicRoughnessIndex = getImageIndex(
                gltfData, pbrParams.metallic_roughness_texture.texture);
            if (metallicRoughnessIndex.has_value())
            {
                if (sRgbColorImages.contains(*metallicRoughnessIndex))
                    printImageColorSpaceReuseWarning(
                        gltfData->textures[*metallicRoughnessIndex].image);
                else
                    linearColorImages.insert(*metallicRoughnessIndex);
            }
        }

        const Optional<uint32_t> normalIndex =
            getImageIndex(gltfData, material.normal_texture.texture);
        if (normalIndex.has_value())
        {
            if (sRgbColorImages.contains(*normalIndex))
                printImageColorSpaceReuseWarning(
                    gltfData->textures[*normalIndex].image);
            else
                linearColorImages.insert(*normalIndex);
        }
    }

    worker = std::thread{&loadingWorker, this};
}

void DeferredLoadingContext::kill()
{
    // This is ok to call unconditionally even if init() hasn't been called
    if (worker.has_value())
    {
        interruptLoading = true;
        worker->join();
        worker.reset();
    }
}

UploadedGeometryData DeferredLoadingContext::uploadGeometryData(
    const MeshCacheHeader &cacheHeader, const Array<uint8_t> &dataBlob,
    const String &meshName)
{
    WHEELS_ASSERT(initialized);
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

    uint32_t *dstPtr = static_cast<uint32_t *>(geometryUploadBuffer.mapped);
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
                    cacheHeader.tangentsOffset == 0xFFFF'FFFF
                        ? 0xFFFF'FFFF
                        : startOffsetU32 + cacheHeader.tangentsOffset,
                .texCoord0sOffset =
                    cacheHeader.texCoord0sOffset == 0xFFFF'FFFF
                        ? 0xFFFF'FFFF
                        : startOffsetU32 + cacheHeader.texCoord0sOffset,
                .meshletsOffset = startOffsetU32 + cacheHeader.meshletsOffset,
                .meshletBoundsOffset =
                    startOffsetU32 + cacheHeader.meshletBoundsOffset,
                .meshletVerticesOffset =
                    (cacheHeader.usesShortIndices == 1 ? startOffsetU16
                                                       : startOffsetU32) +
                    cacheHeader.meshletVerticesOffset,
                .meshletTrianglesByteOffset =
                    startByteOffset + cacheHeader.meshletTrianglesByteOffset,
                .usesShortIndices = cacheHeader.usesShortIndices,
            },
        .byteOffset = startByteOffset,
        .byteCount = cacheHeader.blobByteCount,
        .meshName = meshName,
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
        Buffer buffer = gDevice.createBuffer(BufferCreateInfo{
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
            const std::lock_guard lock{geometryBuffersMutex};
            geometryBuffers.push_back(WHEELS_MOV(buffer));
        }
        geometryBufferRemainingByteCounts.push_back(sGeometryBufferSize);
    }

    return dstBufferI;
}
