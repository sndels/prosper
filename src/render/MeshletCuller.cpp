#include "MeshletCuller.hpp"

#include "../gfx/Device.hpp"
#include "../scene/Camera.hpp"
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

const uint32_t sArgumentsByteSize = static_cast<uint32_t>(3 * sizeof(uint32_t));
const uint32_t sGeneratorGroupSize = 16;
const uint32_t sCullerGroupSize = 64;

// Keep this a tight upper bound or make arrays dynamic if usage varies a
// lot based on content
const uint32_t sMaxRecordsPerFrame = 2;

enum GeneratorBindingSet : uint32_t
{
    GeneratorGeometryBindingSet,
    GeneratorSceneInstancesBindingSet,
    GeneratorMaterialDatasBindingSet,
    GeneratorMaterialTexturesBindingSet,
    GeneratorStorageBindingSet,
    GeneratorBindingSetCount,
};

struct GeneratorPCBlock
{
    uint matchTransparents;
};

enum CullerBindingSet : uint32_t
{
    CullerCameraBindingSet,
    CullerGeometryBindingSet,
    CullerSceneInstancesBindingSet,
    CullerStorageBindingSet,
    CullerBindingSetCount,
};

ComputePass::Shader generatorDefinitionCallback(
    Allocator &alloc, const WorldDSLayouts &worldDSLayouts)
{
    const size_t len = 168;
    String defines{alloc, len};
    appendDefineStr(defines, "GEOMETRY_SET", GeneratorGeometryBindingSet);
    appendDefineStr(
        defines, "SCENE_INSTANCES_SET", GeneratorSceneInstancesBindingSet);
    appendDefineStr(
        defines, "MATERIAL_DATAS_SET", GeneratorMaterialDatasBindingSet);
    appendDefineStr(
        defines, "MATERIAL_TEXTURES_SET", GeneratorMaterialTexturesBindingSet);
    appendDefineStr(
        defines, "NUM_MATERIAL_SAMPLERS", worldDSLayouts.materialSamplerCount);
    appendDefineStr(defines, "STORAGE_SET", GeneratorStorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/draw_list_generator.comp",
        .debugName = String{alloc, "DrawListGeneratorCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = uvec3{sGeneratorGroupSize, 1, 1},
    };
}

StaticArray<vk::DescriptorSetLayout, GeneratorBindingSetCount - 1>
generatorExternalDsLayouts(const WorldDSLayouts &worldDsLayouts)
{
    StaticArray<vk::DescriptorSetLayout, GeneratorBindingSetCount - 1>
        setLayouts{VK_NULL_HANDLE};
    setLayouts[GeneratorGeometryBindingSet] = worldDsLayouts.geometry;
    setLayouts[GeneratorSceneInstancesBindingSet] =
        worldDsLayouts.sceneInstances;
    setLayouts[GeneratorMaterialDatasBindingSet] = worldDsLayouts.materialDatas;
    setLayouts[GeneratorMaterialTexturesBindingSet] =
        worldDsLayouts.materialTextures;
    return setLayouts;
}

ComputePass::Shader argumentsWriterDefinitionCallback(Allocator &alloc)
{
    const size_t len = 29;
    String defines{alloc, len};
    appendDefineStr(defines, "CULLER_GROUP_SIZE", sCullerGroupSize);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/draw_list_culler_arg_writer.comp",
        .debugName = String{alloc, "DrawListCullerArgWriterCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = uvec3{1, 1, 1},
    };
}

ComputePass::Shader cullerDefinitionCallback(Allocator &alloc)
{
    const size_t len = 96;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CullerCameraBindingSet);
    appendDefineStr(defines, "GEOMETRY_SET", CullerGeometryBindingSet);
    appendDefineStr(
        defines, "SCENE_INSTANCES_SET", CullerSceneInstancesBindingSet);
    appendDefineStr(defines, "STORAGE_SET", CullerStorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/draw_list_culler.comp",
        .debugName = String{alloc, "DrawListCullerCS"},
        .defines = WHEELS_MOV(defines),
        .groupSize = uvec3{sCullerGroupSize, 1, 1},
    };
}

StaticArray<vk::DescriptorSetLayout, CullerBindingSetCount - 1>
cullerExternalDsLayouts(
    const WorldDSLayouts &worldDsLayouts, vk::DescriptorSetLayout camDsLayout)
{
    StaticArray<vk::DescriptorSetLayout, CullerBindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CullerCameraBindingSet] = camDsLayout;
    setLayouts[CullerGeometryBindingSet] = worldDsLayouts.geometry;
    setLayouts[CullerSceneInstancesBindingSet] = worldDsLayouts.sceneInstances;
    return setLayouts;
}

} // namespace

void MeshletCuller::init(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const WorldDSLayouts &worldDsLayouts, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!_initialized);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(resources != nullptr);

    _device = device;
    _resources = resources;
    _drawListGenerator.init(
        scopeAlloc.child_scope(), device, staticDescriptorsAlloc,
        [&worldDsLayouts](Allocator &alloc)
        { return generatorDefinitionCallback(alloc, worldDsLayouts); },
        ComputePassOptions{
            .storageSetIndex = GeneratorStorageBindingSet,
            .perFrameRecordLimit = sMaxRecordsPerFrame,
            .externalDsLayouts = generatorExternalDsLayouts(worldDsLayouts),
        });
    _cullerArgumentsWriter.init(
        scopeAlloc.child_scope(), device, staticDescriptorsAlloc,
        argumentsWriterDefinitionCallback,
        ComputePassOptions{
            .perFrameRecordLimit = sMaxRecordsPerFrame,
        });
    _drawListCuller.init(
        WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
        cullerDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = CullerStorageBindingSet,
            .perFrameRecordLimit = sMaxRecordsPerFrame,
            .externalDsLayouts =
                cullerExternalDsLayouts(worldDsLayouts, camDsLayout),
        });

    _initialized = true;
}

void MeshletCuller::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const WorldDSLayouts &worldDsLayouts, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(_initialized);

    _drawListGenerator.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        [&worldDsLayouts](Allocator &alloc)
        { return generatorDefinitionCallback(alloc, worldDsLayouts); },
        generatorExternalDsLayouts(worldDsLayouts));
    _cullerArgumentsWriter.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        argumentsWriterDefinitionCallback);
    _drawListCuller.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, cullerDefinitionCallback,
        cullerExternalDsLayouts(worldDsLayouts, camDsLayout));
}

void MeshletCuller::startFrame()
{
    WHEELS_ASSERT(_initialized);

    _drawListGenerator.startFrame();
    _cullerArgumentsWriter.startFrame();
    _drawListCuller.startFrame();
}

MeshletCullerOutput MeshletCuller::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, const Camera &cam, uint32_t nextFrame,
    const char *debugPrefix, SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);

    const BufferHandle initialList = recordGenerateList(
        scopeAlloc.child_scope(), cb, mode, world, nextFrame, debugPrefix,
        sceneStats, profiler);

    const BufferHandle cullerArgs = recordWriteCullerArgs(
        scopeAlloc.child_scope(), cb, nextFrame, initialList, debugPrefix,
        profiler);

    const MeshletCullerOutput culledList = recordCullList(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        CullerInput{
            .dataBuffer = initialList,
            .argumentBuffer = cullerArgs,
        },
        debugPrefix, profiler);

    _resources->buffers.release(initialList);
    _resources->buffers.release(cullerArgs);

    return culledList;
}

BufferHandle MeshletCuller::recordGenerateList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, uint32_t nextFrame, const char *debugPrefix,
    SceneStats *sceneStats, Profiler *profiler)
{
    uint32_t meshletCountUpperBound = 0;
    {
        String scopeName{scopeAlloc};
        scopeName.extend(debugPrefix);
        scopeName.extend("MeshletCullerStats");
        const auto _s = profiler->createCpuScope(scopeName.c_str());

        const Scene &scene = world.currentScene();
        const Span<const Model> models = world.models();
        const Span<const Material> materials = world.materials();
        const Span<const MeshInfo> meshInfos = world.meshInfos();

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
                        meshletCountUpperBound += info.meshletCount;
                    }
                }
            }
        }

        WHEELS_ASSERT(
            meshletCountUpperBound <=
                _device->properties().meshShader.maxMeshWorkGroupCount[0] &&
            "Indirect mesh dispatch group count might not fit in the "
            "supported mesh work group count");
    }

    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("MeshletDrawList");

    const uint32_t drawListByteSize =
        static_cast<uint32_t>(sizeof(uint32_t)) +
        meshletCountUpperBound * 2u * static_cast<uint32_t>(sizeof(uint32_t));

    const BufferHandle ret = _resources->buffers.create(
        BufferDescription{
            .byteSize = drawListByteSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        dataName.c_str());

    _drawListGenerator.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{DescriptorInfo{
            vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(ret),
                .range = VK_WHOLE_SIZE,
            },
        }});

    _resources->buffers.transition(cb, ret, BufferState::TransferDst);

    // Clear count as it will be used for atomic adds
    cb.fillBuffer(
        _resources->buffers.nativeHandle(ret), 0, sizeof(uint32_t), 0u);

    _resources->buffers.transition(
        cb, ret, BufferState::ComputeShaderReadWrite);

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("DrawListGeneration");
    const auto _s = profiler->createCpuGpuScope(cb, scopeName.c_str());

    const GeneratorPCBlock pcBlock{
        .matchTransparents = mode == Mode::Transparent ? 1u : 0u,
    };

    const Scene &scene = world.currentScene();
    const WorldDescriptorSets &worldDSes = world.descriptorSets();
    const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, GeneratorBindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[GeneratorGeometryBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[GeneratorSceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[GeneratorMaterialDatasBindingSet] =
        worldDSes.materialDatas[nextFrame];
    descriptorSets[GeneratorMaterialTexturesBindingSet] =
        worldDSes.materialTextures;
    descriptorSets[GeneratorStorageBindingSet] =
        _drawListGenerator.storageSet(nextFrame);

    const StaticArray dynamicOffsets{{
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
        worldByteOffsets.globalMaterialConstants,
    }};

    // We want group per instance so multiply the extent by thread count
    const uvec3 extent =
        glm::uvec3{scene.drawInstanceCount * sGeneratorGroupSize, 1u, 1u};

    _drawListGenerator.record(
        cb, pcBlock, extent, descriptorSets, dynamicOffsets);

    return ret;
}

BufferHandle MeshletCuller::recordWriteCullerArgs(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame,
    BufferHandle drawList, const char *debugPrefix, Profiler *profiler)
{
    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("DrawListCullerArguments");

    const BufferHandle ret = _resources->buffers.create(
        BufferDescription{
            .byteSize = sArgumentsByteSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eIndirectBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        argumentsName.c_str());

    _cullerArgumentsWriter.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(drawList),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(ret),
                .range = VK_WHOLE_SIZE,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), *_resources, cb,
        Transitions{
            .buffers = StaticArray<BufferTransition, 2>{{
                {drawList, BufferState::ComputeShaderRead},
                {ret, BufferState::ComputeShaderWrite},
            }},
        });

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("DrawListCullerArgs");
    const auto _s = profiler->createCpuGpuScope(cb, scopeName.c_str());

    const vk::DescriptorSet ds = _cullerArgumentsWriter.storageSet(nextFrame);

    _cullerArgumentsWriter.record(cb, glm::uvec3{1, 1, 1}, Span{&ds, 1});

    return ret;
}

MeshletCullerOutput MeshletCuller::recordCullList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, uint32_t nextFrame, const CullerInput &input,
    const char *debugPrefix, Profiler *profiler)
{
    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("CulledMeshletDrawList");

    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("MeshDiscpatchArguments");

    const vk::DeviceSize drawListByteSize =
        _resources->buffers.resource(input.dataBuffer).byteSize;
    const MeshletCullerOutput ret{
        .dataBuffer = _resources->buffers.create(
            BufferDescription{
                .byteSize = drawListByteSize,
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

    _drawListCuller.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _resources->buffers.nativeHandle(input.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
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
            .buffers = StaticArray<BufferTransition, 4>{{
                {input.dataBuffer, BufferState::ComputeShaderRead},
                {input.argumentBuffer, BufferState::DrawIndirectRead},
                {ret.dataBuffer, BufferState::ComputeShaderWrite},
                {ret.argumentBuffer, BufferState::ComputeShaderReadWrite},
            }},
        });

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("DrawListCuller");
    const auto _s = profiler->createCpuGpuScope(cb, scopeName.c_str());

    const Scene &scene = world.currentScene();
    const WorldDescriptorSets &worldDSes = world.descriptorSets();
    const WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, CullerBindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[CullerCameraBindingSet] = cam.descriptorSet();
    descriptorSets[CullerGeometryBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[CullerSceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[CullerStorageBindingSet] =
        _drawListCuller.storageSet(nextFrame);

    const StaticArray dynamicOffsets{{
        cam.bufferOffset(),
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
    }};

    const vk::Buffer argumentsHandle =
        _resources->buffers.nativeHandle(input.argumentBuffer);
    _drawListCuller.record(cb, argumentsHandle, descriptorSets, dynamicOffsets);

    return ret;
}
