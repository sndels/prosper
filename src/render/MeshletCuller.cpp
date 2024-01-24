#include "MeshletCuller.hpp"

#include "../gfx/Device.hpp"
#include "../scene/Material.hpp"
#include "../scene/Mesh.hpp"
#include "../scene/Model.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/SceneStats.hpp"
#include "RenderResources.hpp"

using namespace wheels;

namespace
{

// Should be plenty for any scene that's realistically loaded in
const uint32_t sMeshDrawListByteSize = static_cast<uint32_t>(megabytes(5));
const uint32_t sArgumentsByteSize = static_cast<uint32_t>(3 * sizeof(uint32_t));

} // namespace

MeshletCuller::MeshletCuller(Device *device, RenderResources *resources)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);
}

MeshletCuller::~MeshletCuller()
{
    if (_device != nullptr)
    {
        for (auto &clientBuffers : _dataBuffers)
            for (const Buffer &buffer : clientBuffers)
                _device->destroy(buffer);

        for (auto &clientBuffers : _argumentBuffers)
            for (const Buffer &buffer : clientBuffers)
                _device->destroy(buffer);
    }
}

void MeshletCuller::startFrame() { _currentFrameRecordCount = 0; }

MeshletCullerOutput MeshletCuller::record(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, const Camera &cam, uint32_t nextFrame,
    const char *debugPrefix, SceneStats *sceneStats, Profiler *profiler)
{

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("MeshletCuller");
    const auto _s = profiler->createCpuGpuScope(cb, scopeName.c_str(), true);

    String dataUploadName{scopeAlloc};
    dataUploadName.extend(debugPrefix);
    dataUploadName.extend("MeshletDataUpload");

    String argumentsUploadName{scopeAlloc};
    argumentsUploadName.extend(debugPrefix);
    argumentsUploadName.extend("MeshShaderArgumentUpload");

    if (_currentFrameRecordCount >= _dataBuffers[nextFrame].size())
    {
        _dataBuffers[nextFrame].push_back(
            _device->createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize = sMeshDrawListByteSize,
                        .usage = vk::BufferUsageFlagBits::eTransferSrc,
                        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent,
                    },
                .debugName = dataUploadName.c_str(),
            }));
        _argumentBuffers[nextFrame].push_back(
            _device->createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize = sArgumentsByteSize,
                        .usage = vk::BufferUsageFlagBits::eTransferSrc,
                        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent,
                    },
                .debugName = argumentsUploadName.c_str(),
            }));
    }

    const Buffer &dataBuffer =
        _dataBuffers[nextFrame][_currentFrameRecordCount];
    const Buffer &argumentBuffer =
        _argumentBuffers[nextFrame][_currentFrameRecordCount];

    const Scene &scene = world.currentScene();
    const Span<const Model> models = world.models();
    const Span<const Material> materials = world.materials();
    const Span<const MeshInfo> meshInfos = world.meshInfos();

    uint32_t drawInstanceID = 0;
    uint32_t meshletCount = 0;
    uint32_t *dataPtr = reinterpret_cast<uint32_t *>(dataBuffer.mapped);
    for (const auto &instance : scene.modelInstances)
    {
        const auto &model = models[instance.modelID];
        for (const auto &subModel : model.subModels)
        {
            const auto &material = materials[subModel.materialID];
            const auto &info = meshInfos[subModel.meshID];
            // 0 means invalid or not yet loaded
            if (info.indexCount > 0)
            {
                bool shouldDraw =
                    mode == Mode::Opaque
                        ? material.alphaMode != Material::AlphaMode::Blend
                        : material.alphaMode == Material::AlphaMode::Blend;

                if (shouldDraw)
                {
                    for (uint32_t i = 0; i < info.meshletCount; ++i)
                    {
                        *dataPtr++ = drawInstanceID;
                        *dataPtr++ = i;
                    }
                    meshletCount += info.meshletCount;

                    sceneStats->totalMeshCount++;
                    sceneStats->totalTriangleCount += info.indexCount / 3;
                    sceneStats->totalMeshletCount += info.meshletCount;
                    meshletCount += info.meshletCount;
                }
            }
            drawInstanceID++;
        }

        WHEELS_ASSERT(
            meshletCount <=
                _device->properties().meshShader.maxMeshWorkGroupCount[0] &&
            "Indirect mesh dispatch group count might not fit in the "
            "supported mesh work group count");
    }

    uint32_t *argsPtr = reinterpret_cast<uint32_t *>(argumentBuffer.mapped);
    argsPtr[0] = meshletCount;
    argsPtr[1] = 1;
    argsPtr[2] = 1;

    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("MeshDrawList");

    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("MeshDiscpatchArguments");

    const MeshletCullerOutput ret{
        .dataBuffer = _resources->buffers.create(
            BufferDescription{
                .byteSize = sMeshDrawListByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            dataName.c_str()),
        .argumentBuffer = _resources->buffers.create(
            BufferDescription{
                .byteSize = sArgumentsByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eIndirectBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            argumentsName.c_str()),
    };

    const StaticArray barriers{{
        *_resources->buffers.transitionBarrier(
            ret.dataBuffer, BufferState::TransferDst, true),
        *_resources->buffers.transitionBarrier(
            ret.argumentBuffer, BufferState::TransferDst, true),
    }};
    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
    });

    {
        const vk::BufferCopy region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sMeshDrawListByteSize,
        };
        cb.copyBuffer(
            dataBuffer.handle, _resources->buffers.nativeHandle(ret.dataBuffer),
            1, &region);
    }

    {
        const vk::BufferCopy region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sArgumentsByteSize,
        };
        cb.copyBuffer(
            argumentBuffer.handle,
            _resources->buffers.nativeHandle(ret.argumentBuffer), 1, &region);
    }

    _currentFrameRecordCount++;

    return ret;
}
