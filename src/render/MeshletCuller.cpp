#include "MeshletCuller.hpp"

#include "gfx/Device.hpp"
#include "render/DrawStats.hpp"
#include "render/RenderResourceHandle.hpp"
#include "render/RenderResources.hpp"
#include "scene/Camera.hpp"
#include "scene/Material.hpp"
#include "scene/Mesh.hpp"
#include "scene/Model.hpp"
#include "scene/Scene.hpp"
#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"

#include <shader_structs/push_constants/draw_list_culler.h>
#include <shader_structs/push_constants/draw_list_generator.h>

using namespace glm;
using namespace wheels;

namespace render
{

namespace
{

const uint32_t sArgumentsByteSize = static_cast<uint32_t>(3 * sizeof(uint32_t));
const uint32_t sGeneratorGroupSize = 16;
const uint32_t sCullerGroupSize = 64;

// Keep this a tight upper bound or make arrays dynamic if usage varies a
// lot based on content
const uint32_t sMaxRecordsPerFrame = 2;

const uint32_t sMaxHierarchicalDepthMips = 12;

enum GeneratorBindingSet : uint8_t
{
    GeneratorGeometryBindingSet,
    GeneratorSceneInstancesBindingSet,
    GeneratorMaterialDatasBindingSet,
    GeneratorMaterialTexturesBindingSet,
    GeneratorStorageBindingSet,
    GeneratorBindingSetCount,
};

enum CullerBindingSet : uint8_t
{
    CullerCameraBindingSet,
    CullerGeometryBindingSet,
    CullerSceneInstancesBindingSet,
    CullerStorageBindingSet,
    CullerBindingSetCount,
};

ComputePass::Shader generatorDefinitionCallback(
    Allocator &alloc, const scene::WorldDSLayouts &worldDSLayouts)
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
generatorExternalDsLayouts(const scene::WorldDSLayouts &worldDsLayouts)
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
    const scene::WorldDSLayouts &worldDsLayouts,
    vk::DescriptorSetLayout camDsLayout)
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
    ScopedScratch scopeAlloc, const scene::WorldDSLayouts &worldDsLayouts,
    vk::DescriptorSetLayout camDsLayout)
{
    WHEELS_ASSERT(!m_initialized);

    m_drawListGenerator.init(
        scopeAlloc.child_scope(), [&worldDsLayouts](Allocator &alloc)
        { return generatorDefinitionCallback(alloc, worldDsLayouts); },
        ComputePassOptions{
            .storageSetIndex = GeneratorStorageBindingSet,
            .storageSetInstanceCount = sMaxRecordsPerFrame,
            .externalDsLayouts = generatorExternalDsLayouts(worldDsLayouts),
        });
    m_cullerArgumentsWriter.init(
        scopeAlloc.child_scope(), argumentsWriterDefinitionCallback,
        ComputePassOptions{
            // Twice the records of for two-phase culling
            .storageSetInstanceCount = sMaxRecordsPerFrame * 2,
        });
    m_drawListCuller.init(
        WHEELS_MOV(scopeAlloc), cullerDefinitionCallback,
        ComputePassOptions{
            .storageSetIndex = CullerStorageBindingSet,
            // Twice the records of for two-phase culling
            .storageSetInstanceCount = sMaxRecordsPerFrame * 2,
            .externalDsLayouts =
                cullerExternalDsLayouts(worldDsLayouts, camDsLayout),
        });

    m_initialized = true;
}

void MeshletCuller::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const scene::WorldDSLayouts &worldDsLayouts,
    vk::DescriptorSetLayout camDsLayout)
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

MeshletCullerFirstPhaseOutput MeshletCuller::recordFirstPhase(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const scene::World &world, const scene::Camera &cam, uint32_t nextFrame,
    const Optional<ImageHandle> &inHierarchicalDepth, StrSpan debugPrefix,
    DrawStats &drawStats)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "  DrawListFirstPhase");

    const BufferHandle initialList = recordGenerateList(
        scopeAlloc.child_scope(), cb, mode, world, nextFrame, debugPrefix,
        drawStats);

    const BufferHandle cullerArgs = recordWriteCullerArgs(
        scopeAlloc.child_scope(), cb, nextFrame, initialList, debugPrefix);

    const bool outputSecondPhaseInput = inHierarchicalDepth.has_value();
    const CullOutput culledList = recordCullList(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        CullInput{
            .dataBuffer = initialList,
            .argumentBuffer = cullerArgs,
            .hierarchicalDepth = inHierarchicalDepth,
        },
        outputSecondPhaseInput, debugPrefix);

    gRenderResources.buffers->release(initialList);
    gRenderResources.buffers->release(cullerArgs);

    MeshletCullerFirstPhaseOutput ret{
        .dataBuffer = culledList.dataBuffer,
        .argumentBuffer = culledList.argumentBuffer,
        .secondPhaseInput = culledList.secondPhaseInput,
    };

    return ret;
}

MeshletCullerSecondPhaseOutput MeshletCuller::recordSecondPhase(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::World &world,
    const scene::Camera &cam, uint32_t nextFrame, BufferHandle inputBuffer,
    ImageHandle inHierarchicalDepth, StrSpan debugPrefix)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_GPU_SCOPE(cb, "  DrawListSecondPhase")

    String argsPrefix{scopeAlloc};
    argsPrefix.extend(debugPrefix);
    argsPrefix.extend("SecondPhase");

    const BufferHandle argumentBuffer = recordWriteCullerArgs(
        scopeAlloc.child_scope(), cb, nextFrame, inputBuffer, argsPrefix);

    const CullOutput culledList = recordCullList(
        WHEELS_MOV(scopeAlloc), cb, world, cam, nextFrame,
        CullInput{
            .dataBuffer = inputBuffer,
            .argumentBuffer = argumentBuffer,
            .hierarchicalDepth = inHierarchicalDepth,
        },
        false, debugPrefix);

    gRenderResources.buffers->release(argumentBuffer);

    MeshletCullerSecondPhaseOutput ret{
        .dataBuffer = culledList.dataBuffer,
        .argumentBuffer = culledList.argumentBuffer,
    };

    return ret;
}

BufferHandle MeshletCuller::recordGenerateList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
    const scene::World &world, uint32_t nextFrame, StrSpan debugPrefix,
    DrawStats &drawStats)
{
    uint32_t meshletCountUpperBound = 0;
    {
        const scene::Scene &scene = world.currentScene();
        const Span<const scene::Model> models = world.models();
        const Span<const scene::shader_structs::MaterialData> materials =
            world.materials();
        const Span<const scene::MeshInfo> meshInfos = world.meshInfos();

        for (const scene::ModelInstance &instance : scene.modelInstances)
        {
            bool modelDrawn = false;
            const scene::Model &model = models[instance.modelIndex];
            for (const scene::Model::SubModel &subModel : model.subModels)
            {
                const scene::shader_structs::MaterialData &material =
                    materials[subModel.materialIndex];
                const scene::MeshInfo &info = meshInfos[subModel.meshIndex];
                // 0 means invalid or not yet loaded
                if (info.indexCount > 0)
                {
                    const bool shouldDraw =
                        mode == Mode::Opaque
                            ? material.alphaMode !=
                                  scene::shader_structs::AlphaMode_Blend
                            : material.alphaMode ==
                                  scene::shader_structs::AlphaMode_Blend;

                    if (shouldDraw)
                    {
                        drawStats.totalMeshCount++;
                        drawStats.totalTriangleCount += info.indexCount / 3;
                        drawStats.totalMeshletCount += info.meshletCount;
                        meshletCountUpperBound += info.meshletCount;
                        if (!modelDrawn)
                        {
                            drawStats.totalModelCount++;
                            modelDrawn = true;
                        }
                    }
                }
            }
        }

        WHEELS_ASSERT(
            meshletCountUpperBound <=
                gfx::gDevice.properties().meshShader.maxMeshWorkGroupCount[0] &&
            "Indirect mesh dispatch group count might not fit in the "
            "supported mesh work group count");
    }

    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    dataName.extend("MeshletDrawList");

    const uint32_t drawListByteSize =
        static_cast<uint32_t>(sizeof(uint32_t)) +
        (meshletCountUpperBound * 2u * static_cast<uint32_t>(sizeof(uint32_t)));

    const BufferHandle ret = gRenderResources.buffers->create(
        gfx::BufferDescription{
            .byteSize = drawListByteSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        dataName.c_str());

    const vk::DescriptorSet storageSet = m_drawListGenerator.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{gfx::DescriptorInfo{
            vk::DescriptorBufferInfo{
                .buffer = gRenderResources.buffers->nativeHandle(ret),
                .range = VK_WHOLE_SIZE,
            },
        }});

    gRenderResources.buffers->transition(
        cb, ret, gfx::BufferState::TransferDst);

    // Clear count as it will be used for atomic adds
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(ret), 0, sizeof(uint32_t), 0u);

    gRenderResources.buffers->transition(
        cb, ret, gfx::BufferState::ComputeShaderReadWrite);

    const DrawListGeneratorPC pcBlock{
        .matchTransparents = mode == Mode::Transparent ? 1u : 0u,
    };

    const scene::Scene &scene = world.currentScene();
    const scene::WorldDescriptorSets &worldDSes = world.descriptorSets();
    const scene::WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, GeneratorBindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[GeneratorGeometryBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[GeneratorSceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[GeneratorMaterialDatasBindingSet] =
        worldDSes.materialDatas[nextFrame];
    descriptorSets[GeneratorMaterialTexturesBindingSet] =
        worldDSes.materialTextures;
    descriptorSets[GeneratorStorageBindingSet] = storageSet;

    const StaticArray dynamicOffsets{{
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
        worldByteOffsets.globalMaterialConstants,
    }};

    const uvec3 groupCount{scene.drawInstanceCount, 1u, 1u};
    m_drawListGenerator.record(
        cb, pcBlock, groupCount, descriptorSets,
        ComputePassOptionalRecordArgs{
            .dynamicOffsets = dynamicOffsets,
        });

    return ret;
}

BufferHandle MeshletCuller::recordWriteCullerArgs(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, uint32_t nextFrame,
    BufferHandle drawList, StrSpan debugPrefix)
{
    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    argumentsName.extend("DrawListCullerArguments");

    const BufferHandle ret = gRenderResources.buffers->create(
        gfx::BufferDescription{
            .byteSize = sArgumentsByteSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eIndirectBuffer,
            .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        },
        argumentsName.c_str());

    const vk::DescriptorSet storageSet =
        m_cullerArgumentsWriter.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = gRenderResources.buffers->nativeHandle(drawList),
                    .range = VK_WHOLE_SIZE,
                }},
                gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = gRenderResources.buffers->nativeHandle(ret),
                    .range = VK_WHOLE_SIZE,
                }},
            }});

    transition(
        WHEELS_MOV(scopeAlloc), cb,
        Transitions{
            .buffers = StaticArray<BufferTransition, 2>{{
                {drawList, gfx::BufferState::ComputeShaderRead},
                {ret, gfx::BufferState::ComputeShaderWrite},
            }},
        });

    const uvec3 groupCount{1, 1, 1};
    m_cullerArgumentsWriter.record(cb, groupCount, Span{&storageSet, 1});

    return ret;
}

MeshletCuller::CullOutput MeshletCuller::recordCullList(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::World &world,
    const scene::Camera &cam, uint32_t nextFrame, const CullInput &input,
    bool outputSecondPhaseInput, StrSpan debugPrefix)
{
    String dataName{scopeAlloc};
    dataName.extend(debugPrefix);
    if (outputSecondPhaseInput)
        dataName.extend("FirstPhase");
    // Second phase outputs might be skipped for first phase too so let's not
    // confuse debug naming by adding 'SecondPhase' in that case.
    dataName.extend("CulledMeshletDrawList");

    String secondPhaseDataName{scopeAlloc};
    secondPhaseDataName.extend(debugPrefix);
    secondPhaseDataName.extend("SecondPhaseInputDrawList");

    String argumentsName{scopeAlloc};
    argumentsName.extend(debugPrefix);
    if (outputSecondPhaseInput)
        argumentsName.extend("FirstPhase");
    // Second phase outputs might be skipped for first phase too so let's not
    // confuse debug naming by adding 'SecondPhase' in that case.
    argumentsName.extend("MeshDiscpatchArguments");

    const bool hierarchicalDepthGiven = input.hierarchicalDepth.has_value();
    ImageHandle dummyHierarchicalDepth;
    if (!hierarchicalDepthGiven)
    {
        String dummyHizName{scopeAlloc};
        dummyHizName.extend(debugPrefix);
        dummyHizName.extend("DummyHiZ");

        dummyHierarchicalDepth = gRenderResources.images->create(
            gfx::ImageDescription{
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
    CullOutput ret{
        .dataBuffer = gRenderResources.buffers->create(
            gfx::BufferDescription{
                .byteSize = drawListByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            dataName.c_str()),
        .argumentBuffer = gRenderResources.buffers->create(
            gfx::BufferDescription{
                .byteSize = sArgumentsByteSize,
                .usage = vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eIndirectBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
            argumentsName.c_str()),
        .secondPhaseInput =
            outputSecondPhaseInput
                ? gRenderResources.buffers->create(
                      gfx::BufferDescription{
                          .byteSize = drawListByteSize,
                          .usage = vk::BufferUsageFlagBits::eTransferDst |
                                   vk::BufferUsageFlagBits::eStorageBuffer,
                          .properties =
                              vk::MemoryPropertyFlagBits::eDeviceLocal,
                      },
                      secondPhaseDataName.c_str())
                : Optional<BufferHandle>{},
    };

    // Bind the first buffer pair twice when we don't have hierarchical depth.
    // These binds won't be accessed in the shader
    const BufferHandle secondPhaseDataBindBuffer =
        outputSecondPhaseInput ? *ret.secondPhaseInput : ret.dataBuffer;

    const vk::DescriptorSet storageSet = m_drawListCuller.updateStorageSet(
        scopeAlloc.child_scope(), nextFrame,
        StaticArray{{
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(input.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(ret.dataBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer =
                    gRenderResources.buffers->nativeHandle(ret.argumentBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            gfx::DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = gRenderResources.buffers->nativeHandle(
                    secondPhaseDataBindBuffer),
                .range = VK_WHOLE_SIZE,
            }},
            gfx::DescriptorInfo{hierarchicalDepthInfos},
            gfx::DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = gRenderResources.nearestBorderBlackFloatSampler,
            }},
        }});

    {
        InlineArray<BufferTransition, 3> bufferTransitions;
        bufferTransitions.emplace_back(
            ret.dataBuffer, gfx::BufferState::TransferDst);
        bufferTransitions.emplace_back(
            ret.argumentBuffer, gfx::BufferState::TransferDst);
        if (outputSecondPhaseInput)
            bufferTransitions.emplace_back(
                *ret.secondPhaseInput, gfx::BufferState::TransferDst);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 1>{{
                    {hierarchicalDepth,
                     gfx::ImageState::ComputeShaderSampledRead},
                }},
                .buffers = bufferTransitions,
            });
    }

    // Clear args first as X will be used for atomic adds
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(ret.argumentBuffer), 0,
        sArgumentsByteSize, 0u);
    // Count is also mirrored in data buffer
    cb.fillBuffer(
        gRenderResources.buffers->nativeHandle(ret.dataBuffer), 0,
        sizeof(uint32_t), 0u);
    if (outputSecondPhaseInput)
        // Same goes for count in second phase input
        cb.fillBuffer(
            gRenderResources.buffers->nativeHandle(*ret.secondPhaseInput), 0,
            sizeof(uint32_t), 0u);

    {
        InlineArray<BufferTransition, 5> bufferTransitions;
        bufferTransitions.emplace_back(
            input.dataBuffer, gfx::BufferState::ComputeShaderRead);
        bufferTransitions.emplace_back(
            input.argumentBuffer, gfx::BufferState::DrawIndirectRead);
        bufferTransitions.emplace_back(
            ret.dataBuffer, gfx::BufferState::ComputeShaderReadWrite);
        bufferTransitions.emplace_back(
            ret.argumentBuffer, gfx::BufferState::ComputeShaderReadWrite);
        if (outputSecondPhaseInput)
            bufferTransitions.emplace_back(
                *ret.secondPhaseInput,
                gfx::BufferState::ComputeShaderReadWrite);

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 1>{{
                    {hierarchicalDepth,
                     gfx::ImageState::ComputeShaderSampledRead},
                }},
                .buffers = bufferTransitions,
            });
    }

    const scene::Scene &scene = world.currentScene();
    const scene::WorldDescriptorSets &worldDSes = world.descriptorSets();
    const scene::WorldByteOffsets &worldByteOffsets = world.byteOffsets();

    StaticArray<vk::DescriptorSet, CullerBindingSetCount> descriptorSets{
        VK_NULL_HANDLE};
    descriptorSets[CullerCameraBindingSet] = cam.descriptorSet();
    descriptorSets[CullerGeometryBindingSet] = worldDSes.geometry[nextFrame];
    descriptorSets[CullerSceneInstancesBindingSet] =
        scene.sceneInstancesDescriptorSet;
    descriptorSets[CullerStorageBindingSet] = storageSet;

    const StaticArray dynamicOffsets{{
        cam.bufferOffset(),
        worldByteOffsets.modelInstanceTransforms,
        worldByteOffsets.previousModelInstanceTransforms,
        worldByteOffsets.modelInstanceScales,
    }};

    DrawListCullerPC pcBlock{
        .outputSecondPhaseInput = outputSecondPhaseInput ? 1u : 0u,
    };
    if (input.hierarchicalDepth.has_value())
    {
        const gfx::Image &hizImage =
            gRenderResources.images->resource(*input.hierarchicalDepth);

        pcBlock.hizResolution =
            uvec2{hizImage.extent.width, hizImage.extent.height};
        pcBlock.hizUvScale =
            vec2{cam.resolution()} / (2.f * vec2{pcBlock.hizResolution});
        pcBlock.hizMipCount = hizImage.mipCount;
    }

    const vk::Buffer argumentsHandle =
        gRenderResources.buffers->nativeHandle(input.argumentBuffer);
    m_drawListCuller.record(
        cb, pcBlock, argumentsHandle, descriptorSets,
        ComputePassOptionalRecordArgs{
            .dynamicOffsets = dynamicOffsets,
        });

    if (gRenderResources.images->isValidHandle(dummyHierarchicalDepth))
        gRenderResources.images->release(dummyHierarchicalDepth);

    return ret;
}

} // namespace render
