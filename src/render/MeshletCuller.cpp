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
#include "DrawStats.hpp"
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

const uint32_t sMaxHierarchicalDepthMips = 12;

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

struct CullerPCBlock
{
    // 0 means no hiz bound
    uint hizMipCount{0};
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
    const size_t len = 120;
    String defines{alloc, len};
    appendDefineStr(defines, "CAMERA_SET", CullerCameraBindingSet);
    appendDefineStr(defines, "GEOMETRY_SET", CullerGeometryBindingSet);
    appendDefineStr(
        defines, "SCENE_INSTANCES_SET", CullerSceneInstancesBindingSet);
    appendDefineStr(defines, "STORAGE_SET", CullerStorageBindingSet);
    appendDefineStr(defines, "MAX_HIZ_MIPS", sMaxHierarchicalDepthMips);
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
    ScopedScratch scopeAlloc, const WorldDSLayouts &worldDsLayouts,
    vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_drawListGenerator.init(
        scopeAlloc.child_scope(),
        [&worldDsLayouts](Allocator &alloc)
        { return generatorDefinitionCallback(alloc, worldDsLayouts); },
        ComputePassOptions{
            .storageSetIndex = GeneratorStorageBindingSet,
            .perFrameRecordLimit = sMaxRecordsPerFrame,
            .externalDsLayouts = generatorExternalDsLayouts(worldDsLayouts),
        });
    m_cullerArgumentsWriter.init(
        scopeAlloc.child_scope(), argumentsWriterDefinitionCallback,
        ComputePassOptions{
            .perFrameRecordLimit = sMaxRecordsPerFrame,
        });
    m_drawListCuller.init(
        WHEELS_MOV(scopeAlloc), cullerDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = CullerStorageBindingSet,
            .perFrameRecordLimit = sMaxRecordsPerFrame,
            .externalDsLayouts =
                cullerExternalDsLayouts(worldDsLayouts, camDsLayout),
        });

    m_initialized = true;
}

void MeshletCuller::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const WorldDSLayouts &worldDsLayouts, vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(m_initialized);

    m_drawListGenerator.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        [&worldDsLayouts](Allocator &alloc)
        { return generatorDefinitionCallback(alloc, worldDsLayouts); },
        generatorExternalDsLayouts(worldDsLayouts));
    m_cullerArgumentsWriter.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        argumentsWriterDefinitionCallback);
    m_drawListCuller.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles, cullerDefinitionCallback,
        cullerExternalDsLayouts(worldDsLayouts, camDsLayout));
}

void MeshletCuller::startFrame()
{
    WHEELS_ASSERT(m_initialized);

    m_drawListGenerator.startFrame();
    m_cullerArgumentsWriter.startFrame();
    m_drawListCuller.startFrame();
}

MeshletCullerOutput MeshletCuller::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, const Camera &cam, uint32_t nextFrame,
    Optional<ImageHandle> inHierarchicalDepth, const char *debugPrefix,
    DrawStats *drawStats)
{
    WHEELS_ASSERT(m_initialized);

    String scopeName{scopeAlloc};
    scopeName.extend(debugPrefix);
    scopeName.extend("DrawList");

    PROFILER_CPU_GPU_SCOPE(cb, scopeName.c_str());

    const BufferHandle initialList = recordGenerateList(
        scopeAlloc.child_scope(), cb, mode, world, nextFrame, debugPrefix,
        drawStats);

    const BufferHandle cullerArgs = recordWriteCullerArgs(
        scopeAlloc.child_scope(), cb, nextFrame, initialList, debugPrefix);

    const MeshletCullerOutput culledList = recordCullList(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        CullerInput{
            .dataBuffer = initialList,
            .argumentBuffer = cullerArgs,
            .hierarchicalDepth = inHierarchicalDepth,
        },
        debugPrefix);

    gRenderResources.buffers->release(initialList);
    gRenderResources.buffers->release(cullerArgs);

    return culledList;
}

BufferHandle MeshletCuller::recordGenerateList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const World &world, uint32_t nextFrame, const char *debugPrefix,
    DrawStats *drawStats)
{
    uint32_t meshletCountUpperBound = 0;
    {
        const Scene &scene = world.currentScene();
        const Span<const Model> models = world.models();
        const Span<const Material> materials = world.materials();
        const Span<const MeshInfo> meshInfos = world.meshInfos();

        for (const ModelInstance &instance : scene.modelInstances)
        {
            bool modelDrawn = false;
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
                        drawStats->totalMeshCount++;
                        drawStats->totalTriangleCount += info.indexCount / 3;
                        drawStats->totalMeshletCount += info.meshletCount;
                        meshletCountUpperBound += info.meshletCount;
                        if (!modelDrawn)
                        {
                            drawStats->totalModelCount++;
                            modelDrawn = true;
                        }
                    }
                }
            }
        }

        WHEELS_ASSERT(
            meshletCountUpperBound <=
                gDevice.properties().meshShader.maxMeshWorkGroupCount[0] &&
            "Indirect mesh dispatch group count might not fit in the "
            "supported mesh work group count");
    }

    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("MeshletDrawList");

    const uint32_t drawListByteSize =
        static_cast<uint32_t>(sizeof(uint32_t)) +
        meshletCountUpperBound * 2u * static_cast<uint32_t>(sizeof(uint32_t));

    const BufferHandle ret = gRenderResources.buffers->create(
        BufferDescription{
            .byteSize = drawListByteSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        dataName.c_str());

    m_drawListGenerator.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{DescriptorInfo{
            vk::DescriptorBufferInfo{
                .buffer = gRenderResources.buffers->nativeHandle(ret),
                .range = VK_WHOLE_SIZE,
            },
        }});

    gRenderResources.buffers->transition(cb, ret, BufferState::TransferDst);

    // Clear count as it will be used for atomic adds
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(ret), 0, sizeof(uint32_t), 0u);

    gRenderResources.buffers->transition(
        cb, ret, BufferState::ComputeShaderReadWrite);

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
        m_drawListGenerator.storageSet(nextFrame);

    const StaticArray dynamicOffsets{{
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
        worldByteOffsets.globalMaterialConstants,
    }};

    const uvec3 groupCount{scene.drawInstanceCount, 1u, 1u};
    m_drawListGenerator.record(
        cb, pcBlock, groupCount, descriptorSets, dynamicOffsets);

    return ret;
}

BufferHandle MeshletCuller::recordWriteCullerArgs(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame,
    BufferHandle drawList, const char *debugPrefix)
{
    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("DrawListCullerArguments");

    const BufferHandle ret = gRenderResources.buffers->create(
        BufferDescription{
            .byteSize = sArgumentsByteSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eIndirectBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        argumentsName.c_str());

    m_cullerArgumentsWriter.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = gRenderResources.buffers->nativeHandle(drawList),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = gRenderResources.buffers->nativeHandle(ret),
                .range = VK_WHOLE_SIZE,
            }},
        }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .buffers = StaticArray<BufferTransition, 2>{{
                {drawList, BufferState::ComputeShaderRead},
                {ret, BufferState::ComputeShaderWrite},
            }},
        });

    const vk::DescriptorSet ds = m_cullerArgumentsWriter.storageSet(nextFrame);

    const uvec3 groupCount{1, 1, 1};
    m_cullerArgumentsWriter.record(cb, groupCount, Span{&ds, 1});

    return ret;
}

MeshletCullerOutput MeshletCuller::recordCullList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const World &world,
    const Camera &cam, uint32_t nextFrame, const CullerInput &input,
    const char *debugPrefix)
{
    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("CulledMeshletDrawList");

    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("MeshDiscpatchArguments");

    ImageHandle dummyHierarchicalDepth;
    if (!input.hierarchicalDepth.has_value())
    {
        String dummyHizName{scopeAlloc};
        dummyHizName.extend(debugPrefix);
        dummyHizName.extend("DummyHiZ");

        dummyHierarchicalDepth = gRenderResources.images->create(
            ImageDescription{
                .format = vk::Format::eR32Sfloat,
                .width = 1,
                .height = 1,
                .mipCount = 1,
                .usageFlags = vk::ImageUsageFlagBits::eSampled,
            },
            dummyHizName.c_str());
    }
    // TODO:
    // Just enable null binds instead of binding dummies?
    const ImageHandle hierarchicalDepth = input.hierarchicalDepth.has_value()
                                              ? *input.hierarchicalDepth
                                              : dummyHierarchicalDepth;

    const Span<const vk::ImageView> hierarchicalDepthViews =
        gRenderResources.images->subresourceViews(hierarchicalDepth);

    StaticArray<vk::DescriptorImageInfo, sMaxHierarchicalDepthMips>
        hierarchicalDepthInfos;
    {
        size_t i = 0;
        for (; i < hierarchicalDepthViews.size(); ++i)
            hierarchicalDepthInfos[i] = vk::DescriptorImageInfo{
                .imageView = hierarchicalDepthViews[i],
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
        // Fill the remaining descriptors with copies of the first one so we
        // won't have unbound descriptors. We could use VK_EXT_robustness2 and
        // null descriptors, but this seems like less of a hassle since we
        // shouldn't be accessing them anyway.
        for (; i < sMaxHierarchicalDepthMips; ++i)
            hierarchicalDepthInfos[i] = vk::DescriptorImageInfo{
                .imageView = hierarchicalDepthViews[0],
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
    }

    const vk::DeviceSize drawListByteSize =
        gRenderResources.buffers->resource(input.dataBuffer).byteSize;
    const MeshletCullerOutput ret{
        .dataBuffer = gRenderResources.buffers->create(
            BufferDescription{
                .byteSize = drawListByteSize,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            dataName.c_str()),
        .argumentBuffer = gRenderResources.buffers->create(
            BufferDescription{
                .byteSize = sArgumentsByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eIndirectBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            argumentsName.c_str()),
    };

    m_drawListCuller.updateDescriptorSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(input.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(ret.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(ret.argumentBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{hierarchicalDepthInfos},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestBorderBlackFloatSampler,
            }},
        }});

    gRenderResources.buffers->transition(
        cb, ret.argumentBuffer, BufferState::TransferDst);

    // Clear args first as X will be used for atomic adds
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(ret.argumentBuffer), 0,
        sArgumentsByteSize, 0u);

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .images = StaticArray<ImageTransition, 1>{{
                {hierarchicalDepth, ImageState::ComputeShaderSampledRead},
            }},
            .buffers = StaticArray<BufferTransition, 4>{{
                {input.dataBuffer, BufferState::ComputeShaderRead},
                {input.argumentBuffer, BufferState::DrawIndirectRead},
                {ret.dataBuffer, BufferState::ComputeShaderWrite},
                {ret.argumentBuffer, BufferState::ComputeShaderReadWrite},
            }},
        });

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
        m_drawListCuller.storageSet(nextFrame);

    const StaticArray dynamicOffsets{{
        cam.bufferOffset(),
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
    }};

    const CullerPCBlock pcBlock{
        .hizMipCount =
            input.hierarchicalDepth.has_value()
                ? gRenderResources.images->resource(*input.hierarchicalDepth)
                      .mipCount
                : 0,
    };

    const vk::Buffer argumentsHandle =
        gRenderResources.buffers->nativeHandle(input.argumentBuffer);
    m_drawListCuller.record(
        cb, pcBlock, argumentsHandle, descriptorSets, dynamicOffsets);

    if (gRenderResources.images->isValidHandle(dummyHierarchicalDepth))
        gRenderResources.images->release(dummyHierarchicalDepth);

    return ret;
}
