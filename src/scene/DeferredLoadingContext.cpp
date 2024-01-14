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
void copyBytes(uint32_t *dst, const Array<T> &src, uint32_t dstU32Offset)
{
    if (src.empty())
        return;

    memcpy(dst + dstU32Offset, src.data(), src.size() * sizeof(T));
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

DeferredLoadingContext::MeshData getMeshData(
    Allocator &alloc, const tinygltf::Model &gltfModel,
    const InputGeometryMetadata &metadata, const MeshInfo &meshInfo)
{
    DeferredLoadingContext::MeshData ret{
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

    const DeferredLoadingContext::MeshData *meshData =
        reinterpret_cast<const DeferredLoadingContext::MeshData *>(
            pContext->m_pUserData);

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

    const DeferredLoadingContext::MeshData *meshData =
        reinterpret_cast<const DeferredLoadingContext::MeshData *>(
            pContext->m_pUserData);

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

    const DeferredLoadingContext::MeshData *meshData =
        reinterpret_cast<const DeferredLoadingContext::MeshData *>(
            pContext->m_pUserData);

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

    const DeferredLoadingContext::MeshData *meshData =
        reinterpret_cast<const DeferredLoadingContext::MeshData *>(
            pContext->m_pUserData);

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

    DeferredLoadingContext::MeshData *meshData =
        reinterpret_cast<DeferredLoadingContext::MeshData *>(
            pContext->m_pUserData);

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
void generateTangents(
    Allocator &alloc, DeferredLoadingContext::MeshData *meshData)
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
    Allocator &alloc, DeferredLoadingContext::MeshData *meshData,
    const std::string &meshName)
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

void loadNextMesh(ScopedScratch scopeAlloc, DeferredLoadingContext *ctx)
{
    WHEELS_ASSERT(ctx != nullptr);

    const uint32_t meshIndex = ctx->workerLoadedMeshCount;
    WHEELS_ASSERT(meshIndex < ctx->meshes.size());

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

    DeferredLoadingContext::MeshData meshData =
        getMeshData(scopeAlloc, ctx->gltfModel, metadata, info);

    if (meshData.tangents.empty() && !meshData.texCoord0s.empty())
    {
        generateTangents(scopeAlloc, &meshData);
        info.vertexCount = asserted_cast<uint32_t>(meshData.positions.size());
    }

    optimizeMeshData(
        scopeAlloc, &meshData,
        ctx->gltfModel.meshes[metadata.sourceMeshIndex].name);

    const UploadedGeometryData uploadData = ctx->uploadGeometryData(
        scopeAlloc.child_scope(), WHEELS_MOV(meshData), info);

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

        ctx->loadedMeshes.emplace_back(uploadData, nextMesh.second);
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
    // Make clang-tidy treat WHEELS_ASSERT as assert so that it considers them
    // valid null checks
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
    ScopedScratch scopeAlloc, MeshData &&meshData, const MeshInfo &meshInfo)
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
        // Just use the data directly instead the extra allocation and memcpy?
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

    const uint32_t dstBufferI = getGeometryBuffer(byteCount);

    // The mesh data ranges are expected to not leave gaps in the buffer so that
    // ownership is transferred properly between the queues.
    const uint32_t startByteOffset =
        sGeometryBufferSize - geometryBufferRemainingByteCounts[dstBufferI];

    UploadedGeometryData ret{
        .metadata =
            GeometryMetadata{
                .bufferIndex = dstBufferI,
                .indicesOffset = startByteOffset / elementSize,
                .positionsOffset =
                    (startByteOffset + indicesByteCount) / elementSize,
                .normalsOffset =
                    (startByteOffset + indicesByteCount + positionsByteCount) /
                    elementSize,
                .tangentsOffset =
                    hasTangents ? (startByteOffset + indicesByteCount +
                                   positionsByteCount + normalsByteCount) /
                                      elementSize
                                : 0xFFFFFFFF,
                .texCoord0sOffset =
                    hasTexCoord0s ? (startByteOffset + indicesByteCount +
                                     positionsByteCount + normalsByteCount +
                                     tangentsByteCount) /
                                        elementSize
                                  : 0xFFFFFFFF,
                .usesShortIndices = usesShortIndices},
        .byteCount = byteCount,
    };

    uint32_t *dstPtr =
        reinterpret_cast<uint32_t *>(geometryUploadBuffer.mapped);
    // Let's just write straight into the dst offsets as our upload buffer is as
    // big as the destination buffer
    copyBytes(dstPtr, packedIndices, ret.metadata.indicesOffset);
    copyBytes(dstPtr, meshData.positions, ret.metadata.positionsOffset);
    copyBytes(dstPtr, meshData.normals, ret.metadata.normalsOffset);
    copyBytes(dstPtr, meshData.tangents, ret.metadata.tangentsOffset);
    copyBytes(dstPtr, meshData.texCoord0s, ret.metadata.texCoord0sOffset);

    const vk::BufferCopy copyRegion{
        .srcOffset = startByteOffset,
        .dstOffset = startByteOffset,
        .size = byteCount,
    };
    const Buffer &dstBuffer = geometryBuffers[dstBufferI];
    // Don't use the context command buffer since this method is also used by
    // the non-async loading implementations
    cb.copyBuffer(
        geometryUploadBuffer.handle, dstBuffer.handle, 1, &copyRegion);

    // The mesh data ranges are expected to not leave gaps in the buffer so that
    // ownership is transferred properly between the queues. Any
    // alignment/padding for the next mesh should be included in the byte
    // count of the previous one.
    geometryBufferRemainingByteCounts[dstBufferI] -= byteCount;

    return ret;
}

uint32_t DeferredLoadingContext::getGeometryBuffer(uint32_t byteCount)
{
    // Find a buffer that fits the data or create a new one
    // Let's assume there's only a handful of these so we can just comb through
    // all of them and potentially fill early buffers more completely than if we
    // just checked the last one.
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
