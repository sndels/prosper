#include "MeshletCuller.hpp"

#include "../gfx/Device.hpp"
#include "../scene/Material.hpp"
#include "../scene/Mesh.hpp"
#include "../scene/Model.hpp"
#include "../scene/Scene.hpp"
#include "../scene/World.hpp"
#include "../scene/WorldRenderStructs.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/SceneStats.hpp"
#include "RenderResources.hpp"

using namespace glm;
using namespace wheels;

namespace
{

// Should be plenty for any scene that's realistically loaded in
const uint32_t sMeshDrawListByteSize = static_cast<uint32_t>(megabytes(5));
const uint32_t sArgumentsByteSize = static_cast<uint32_t>(3 * sizeof(uint32_t));
const uint32_t sThreadsPerMesh = 16;

// Keep this a tight upper bound or make arrays dynamic if usage varies a
// lot based on content
const uint32_t sMaxRecordsPerFrame = 2;

enum BindingSet : uint32_t
{
    GeometryBindingSet,
    SceneInstancesBindingSet,
    MaterialDatasBindingSet,
    MaterialTexturesBindingSet,
    StorageBindingSet,
    BindingSetCount,
};

struct PCBlock
{
    uint matchTransparents;
};

ComputePass::Shader shaderDefinitionCallback(
    Allocator &alloc, const WorldDSLayouts &worldDSLayouts)
{
    const size_t len = 168;
    String defines{alloc, len};
    appendDefineStr(defines, "GEOMETRY_SET", GeometryBindingSet);
    appendDefineStr(defines, "SCENE_INSTANCES_SET", SceneInstancesBindingSet);
    appendDefineStr(defines, "MATERIAL_DATAS_SET", MaterialDatasBindingSet);
    appendDefineStr(
        defines, "MATERIAL_TEXTURES_SET", MaterialTexturesBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/draw_list_generator.comp",
        .debugName = String{alloc, "DrawListGeneratorCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = uvec3{sThreadsPerMesh, 1, 1},
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const WorldDSLayouts &worldDsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[GeometryBindingSet] = worldDsLayouts.geometry;
    setLayouts[SceneInstancesBindingSet] = worldDsLayouts.sceneInstances;
    setLayouts[MaterialDatasBindingSet] = worldDsLayouts.materialDatas;
    setLayouts[MaterialTexturesBindingSet] = worldDsLayouts.materialTextures;
    return setLayouts;
}

} // namespace

MeshletCuller::MeshletCuller(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const WorldDSLayouts &worldDsLayouts)
: _device{device}
, _resources{resources}
, _drawListGenerator{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      [&worldDsLayouts](Allocator &alloc)
      { return shaderDefinitionCallback(alloc, worldDsLayouts); },
      ComputePassOptions{
          .storageSetIndex = StorageBindingSet,
          .perFrameRecordLimit = sMaxRecordsPerFrame,
          .externalDsLayouts = externalDsLayouts(worldDsLayouts),
      }}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);
}

void MeshletCuller::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const WorldDSLayouts &worldDsLayouts)
{
    _drawListGenerator.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles,
        [&worldDsLayouts](Allocator &alloc)
        { return shaderDefinitionCallback(alloc, worldDsLayouts); },
        externalDsLayouts(worldDsLayouts));
}

void MeshletCuller::startFrame() { _drawListGenerator.startFrame(); }

MeshletCullerOutput MeshletCuller::record(
    wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, const Camera &cam, uint32_t nextFrame,
    const char *debugPrefix, SceneStats *sceneStats, Profiler *profiler)
{
    {
        String scopeName{scopeAlloc};
        scopeName.extend(debugPrefix);
        scopeName.extend("MeshletCullerStats");
        const auto _s = profiler->createCpuScope(scopeName.c_str());

        const Scene &scene = world.currentScene();
        const Span<const Model> models = world.models();
        const Span<const Material> materials = world.materials();
        const Span<const MeshInfo> meshInfos = world.meshInfos();

        // TODO:
        // Stats from GPU instead? This might go out of sync with gpu draw list
        // code
        uint32_t drawListUpperBound = 0;
        for (const ModelInstance &instance : scene.modelInstances)
        {
            const Model &model = models[instance.modelID];
            for (const Model::SubModel &subModel : model.subModels)
            {
                const Material &material = materials[subModel.materialID];
                const MeshInfo &info = meshInfos[subModel.meshID];
                // 0 means invalid or not yet loaded
                if (info.indexCount > 0)
                {
                    const bool shouldDraw =
                        mode == Mode::Opaque
                            ? material.alphaMode != Material::AlphaMode::Blend
                            : material.alphaMode == Material::AlphaMode::Blend;

                    if (shouldDraw)
                    {
                        sceneStats->totalMeshCount++;
                        sceneStats->totalTriangleCount += info.indexCount / 3;
                        sceneStats->totalMeshletCount += info.meshletCount;
                        drawListUpperBound += info.meshletCount;
                    }
                }
            }
        }

        WHEELS_ASSERT(
            drawListUpperBound <=
                _device->properties().meshShader.maxMeshWorkGroupCount[0] &&
            "Indirect mesh dispatch group count might not fit in the "
            "supported mesh work group count");
    }

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
                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            dataName.c_str()),
        .argumentBuffer = _resources->buffers.create(
            BufferDescription{
                .byteSize = sArgumentsByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eIndirectBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            argumentsName.c_str()),
    };

    _drawListGenerator.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(ret.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(ret.argumentBuffer),
                .range = VK_WHOLE_SIZE,
            }},
        }});

    _resources->buffers.transition(
        cb, ret.argumentBuffer, BufferState::TransferDst);

    // Clear args first as X will be used for atomic adds
    cb.fillBuffer(
        _resources->buffers.nativeHandle(ret.argumentBuffer), 0,
        sArgumentsByteSize, 0u);

    transition(
        WHEELS_MOV(scopeAlloc), *_resources, cb,
        Transitions{
            .buffers = StaticArray<BufferTransition, 2>{{
                {ret.dataBuffer, BufferState::ComputeShaderWrite},
                {ret.argumentBuffer, BufferState::ComputeShaderReadWrite},
            }},
        });

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("DrawListGeneration");
    const auto _s = profiler->createCpuGpuScope(cb, scopeName.c_str());

    const PCBlock pcBlock{
        .matchTransparents = mode == Mode::Transparent ? 1u : 0u,
    };

    const Scene &scene = world.currentScene();
    const WorldDescriptorSets &worldDSes = world.descriptorSets();
    const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[GeometryBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[SceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[MaterialDatasBindingSet] =
        worldDSes.materialDatas[nextFrame];
    descriptorSets[MaterialTexturesBindingSet] = worldDSes.materialTextures;
    descriptorSets[StorageBindingSet] =
        _drawListGenerator.storageSet(nextFrame);

    const StaticArray dynamicOffsets{{
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
        worldByteOffsets.globalMaterialConstants,
    }};

    WHEELS_ASSERT(
        scene.drawInstanceCount * 2u *
            static_cast<uint32_t>(sizeof(uint32_t)) <=
        sMeshDrawListByteSize);

    // We want group per instance so multiply the extent by thread count
    const uvec3 extent =
        glm::uvec3{scene.drawInstanceCount * sThreadsPerMesh, 1u, 1u};

    _drawListGenerator.record(
        cb, pcBlock, extent, descriptorSets, dynamicOffsets);

    return ret;
}
