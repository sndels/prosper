#include "World.hpp"

#include <imgui.h>
#include <memory>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>

#include "../gfx/Device.hpp"
#include "../gfx/RingBuffer.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Ui.hpp"
#include "../utils/Utils.hpp"
#include "Animations.hpp"
#include "Camera.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "WorldData.hpp"

using namespace glm;
using namespace wheels;

namespace
{

std::unique_ptr<RingBuffer> createLightDataRing(Device *device)
{
    WHEELS_ASSERT(device != nullptr);

    const uint32_t bufferSize =
        (DirectionalLight::sBufferByteSize + RingBuffer::sAlignment +
         PointLights::sBufferByteSize + RingBuffer::sAlignment +
         SpotLights::sBufferByteSize + RingBuffer::sAlignment) *
        MAX_FRAMES_IN_FLIGHT;
    return std::make_unique<RingBuffer>(
        device, vk::BufferUsageFlagBits::eStorageBuffer, bufferSize,
        "LightDataRing");
}

} // namespace

// TODO: Split scene loading and runtime scene into separate classes, CUs
class World::Impl
{
  public:
    Impl(
        Allocator &generalAlloc, ScopedScratch scopeAlloc, Device *device,
        RingBuffer *constantsRing, const std::filesystem::path &scene);
    ~Impl();

    Impl(const Impl &other) = delete;
    Impl(Impl &&other) = delete;
    Impl &operator=(const Impl &other) = delete;
    Impl &operator=(Impl &&other) = delete;

    void startFrame();
    void endFrame();

    void uploadMeshDatas(wheels::ScopedScratch scopeAlloc, uint32_t nextFrame);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) one callsite
    void uploadMaterialDatas(uint32_t nextFrame, float lodBias);

    bool drawSceneUi();
    bool drawCameraUi();

    [[nodiscard]] Scene &currentScene();
    [[nodiscard]] const Scene &currentScene() const;
    [[nodiscard]] AccelerationStructure &currentTLAS();
    void updateAnimations(float timeS, Profiler *profiler);
    // Has to be called after updateAnimations()
    void updateScene(
        ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
        Profiler *profiler);
    void updateBuffers(ScopedScratch scopeAlloc);
    // Has to be called after updateBuffers(). Returns true if new BLASes were
    // added.
    bool buildAccelerationStructures(vk::CommandBuffer cb);
    void drawSkybox(vk::CommandBuffer cb) const;

  private:
    Buffer &reserveScratch(vk::DeviceSize byteSize);
    void buildNextBlas(vk::CommandBuffer cb);
    void buildCurrentTlas(vk::CommandBuffer cb);
    void reserveTlasInstances(uint32_t instanceCount);
    [[nodiscard]] AccelerationStructure createTlas(
        const Scene &scene, vk::AccelerationStructureBuildSizesInfoKHR sizeInfo,
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo);
    void updateTlasInstances(ScopedScratch scopeAlloc, const Scene &scene);
    void createTlasBuildInfos(
        const Scene &scene,
        vk::AccelerationStructureBuildRangeInfoKHR &rangeInfoOut,
        vk::AccelerationStructureGeometryKHR &geometryOut,
        vk::AccelerationStructureBuildGeometryInfoKHR &buildInfoOut,
        vk::AccelerationStructureBuildSizesInfoKHR &sizeInfoOut);

    Allocator &_generalAlloc;
    RingBuffer *_constantsRing;
    Device *_device{nullptr};
    std::unique_ptr<RingBuffer> _lightDataRing;
    wheels::Optional<size_t> _nextScene;
    uint32_t _framesSinceFinalBlasBuilds{0};
    Timer _blasBuildTimer;

  public:
    WorldData _data;
    uint32_t _currentCamera{0};

    WorldByteOffsets _byteOffsets;

    struct ScratchBuffer
    {
        uint32_t framesSinceLastUsed{0};
        Buffer buffer;
    };
    Array<ScratchBuffer> _scratchBuffers{_generalAlloc};
    Buffer _tlasInstancesBuffer;
    std::unique_ptr<RingBuffer> _tlasInstancesUploadRing;
    uint32_t _tlasInstancesUploadOffset{0};
};

World::Impl::Impl(
    Allocator &generalAlloc, ScopedScratch scopeAlloc, Device *device,
    RingBuffer *constantsRing, const std::filesystem::path &scene)
: _generalAlloc{generalAlloc}
, _constantsRing{constantsRing}
, _device{device}
, _lightDataRing{createLightDataRing(_device)}
, _data{
      generalAlloc, WHEELS_MOV(scopeAlloc), device,
      WorldData::RingBuffers{
          .constantsRing = _constantsRing,
          .lightDataRing = _lightDataRing.get(),
      },
      scene}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_constantsRing != nullptr);

    // This creates the instance ring and startFrame() assumes it exists
    reserveTlasInstances(1);
}

World::Impl::~Impl()
{
    for (auto &sb : _scratchBuffers)
        _device->destroy(sb.buffer);
    _device->destroy(_tlasInstancesBuffer);
}

void World::Impl::startFrame()
{
    // Launch on the first frame instead of the WorldData ctor to avoid the
    // deferred loading timer bloating from renderer setup etc.
    if (_data._deferredLoadingContext.has_value() &&
        !_data._deferredLoadingContext->worker.has_value())
        _data._deferredLoadingContext->launch();

    if (_nextScene.has_value())
    {
        {
            Scene &scene = currentScene();
            // Transforms will be invalid the next time we select the current
            // scene
            for (ModelInstance &mi : scene.modelInstances)
                mi.previousTransformValid = false;
        }

        _data._currentScene = _nextScene.take();
    }
    _data._modelInstanceTransformsRing->startFrame();
    _lightDataRing->startFrame();
    _tlasInstancesUploadRing->startFrame();

    for (size_t i = 0; i < _scratchBuffers.size();)
    {
        ScratchBuffer &sb = _scratchBuffers[i];
        // TODO:l
        // Should this free logic be done for al the tracked render resources?
        if (++sb.framesSinceLastUsed > MAX_FRAMES_IN_FLIGHT)
        {
            // No in-flight frames are using the buffer so it can be safely
            // destroyed
            _device->destroy(sb.buffer);
            // The reference held by sb is invalid after this
            _scratchBuffers.erase(i);
        }
        else
            ++i;
    }
}

void World::Impl::endFrame()
{
    Scene &scene = currentScene();
    for (ModelInstance &mi : scene.modelInstances)
        mi.previousTransformValid = true;
}

void World::Impl::uploadMeshDatas(
    wheels::ScopedScratch scopeAlloc, uint32_t nextFrame)
{
    _data.uploadMeshDatas(WHEELS_MOV(scopeAlloc), nextFrame);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) one callsite
void World::Impl::uploadMaterialDatas(uint32_t nextFrame, float lodBias)
{
    _data.uploadMaterialDatas(nextFrame);

    _byteOffsets.globalMaterialConstants = _constantsRing->write_value(lodBias);
}

bool World::Impl::drawSceneUi()
{
    WHEELS_ASSERT(!_data._scenes.empty());

    bool sceneChanged = false;
    if (_data._scenes.size() > 1)
    {
        ImGui::SetNextWindowPos(ImVec2{60.f, 60.f}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        const uint32_t sceneCount =
            asserted_cast<uint32_t>(_data._scenes.size());
        if (sceneCount > 1)
        {
            uint32_t scene = asserted_cast<uint32_t>(_data._currentScene);
            if (sliderU32("Active scene", &scene, 0, sceneCount - 1))
            {
                // Make sure the new camera's parameters are copied over from
                sceneChanged = true;
                _nextScene = scene;
            }
        }

        ImGui::End();
    }

    return sceneChanged;
}

bool World::Impl::drawCameraUi()
{
    WHEELS_ASSERT(!_data._cameras.empty());
    const uint32_t cameraCount = asserted_cast<uint32_t>(_data._cameras.size());
    bool camChanged = false;
    if (cameraCount > 1)
    {
        if (sliderU32("Active camera", &_currentCamera, 0, cameraCount - 1))
        {
            // Make sure the new camera's parameters are copied over from
            camChanged = true;
        }
    }
    return camChanged;
}

Scene &World::Impl::currentScene()
{
    return _data._scenes[_data._currentScene];
}

const Scene &World::Impl::currentScene() const
{
    return _data._scenes[_data._currentScene];
}

AccelerationStructure &World::Impl::currentTLAS()
{
    return _data._tlases[_data._currentScene];
}

void World::Impl::updateAnimations(float timeS, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    auto _s = profiler->createCpuScope("World::updateAnimations");

    for (Animation<vec3> &animation : _data._animations._vec3)
        animation.update(timeS);
    for (Animation<quat> &animation : _data._animations._quat)
        animation.update(timeS);
}

void World::Impl::updateScene(
    ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
    Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);
    WHEELS_ASSERT(cameraTransform != nullptr);

    auto _s = profiler->createCpuScope("World::updateScene");

    Scene &scene = currentScene();

    Array<uint32_t> nodeStack{scopeAlloc, scene.nodes.size()};
    Array<mat4> parentTransforms{scopeAlloc, scene.nodes.size()};
    HashSet<uint32_t> visited{scopeAlloc, scene.nodes.size()};
    for (uint32_t rootIndex : scene.rootNodes)
    {
        nodeStack.clear();
        parentTransforms.clear();
        visited.clear();

        nodeStack.push_back(rootIndex);
        parentTransforms.push_back(mat4{1.f});
        while (!nodeStack.empty())
        {
            const uint32_t nodeIndex = nodeStack.back();
            if (visited.find(nodeIndex) != visited.end())
            {
                nodeStack.pop_back();
                parentTransforms.pop_back();
            }
            else
            {
                visited.insert(nodeIndex);
                Scene::Node &node = scene.nodes[nodeIndex];

                const uint32_t first_child = node.firstChild;
                const uint32_t last_child = node.lastChild;
                for (uint32_t child = first_child; child <= last_child; ++child)
                    nodeStack.push_back(child);

                mat4 modelToWorld4x4 = parentTransforms.back();
                if (node.translation.has_value())
                    modelToWorld4x4 =
                        translate(modelToWorld4x4, *node.translation);
                if (node.rotation.has_value())
                    modelToWorld4x4 *= mat4_cast(*node.rotation);
                if (node.scale.has_value())
                    modelToWorld4x4 = scale(modelToWorld4x4, *node.scale);

                const mat3x4 modelToWorld = transpose(modelToWorld4x4);
                // No transpose as mat4->mat3x4 effectively does it
                const mat3x4 normalToWorld = inverse(modelToWorld4x4);

                if (node.modelInstance.has_value())
                    scene.modelInstances[*node.modelInstance].transforms =
                        ModelInstance::Transforms{
                            .modelToWorld = modelToWorld,
                            .normalToWorld = normalToWorld,
                        };

                if (node.camera.has_value() && *node.camera == _currentCamera)
                {
                    cameraTransform->eye =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f}};
                    // TODO: Halfway from camera to scene bb end if inside
                    // bb / halfway of bb if outside of bb?
                    cameraTransform->target =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, -1.f, 1.f}};
                    cameraTransform->up =
                        mat3{modelToWorld4x4} * vec3{0.f, 1.f, 0.f};
                }

                if (node.directionalLight)
                {
                    auto &parameters = scene.lights.directionalLight.parameters;
                    parameters.direction =
                        vec4{mat3{modelToWorld4x4} * vec3{0.f, 0.f, -1.f}, 0.f};
                }

                if (node.pointLight.has_value())
                {
                    PointLight &sceneLight =
                        scene.lights.pointLights.data[*node.pointLight];

                    sceneLight.position =
                        modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f};
                }

                if (node.spotLight.has_value())
                {
                    SpotLight &sceneLight =
                        scene.lights.spotLights.data[*node.spotLight];

                    const vec3 position =
                        vec3{modelToWorld4x4 * vec4{0.f, 0.f, 0.f, 1.f}};
                    sceneLight.positionAndAngleOffset.x = position.x;
                    sceneLight.positionAndAngleOffset.y = position.y;
                    sceneLight.positionAndAngleOffset.z = position.z;

                    sceneLight.direction =
                        vec4{mat3{modelToWorld4x4} * vec3{0.f, 0.f, -1.f}, 0.f};
                }
                parentTransforms.emplace_back(modelToWorld4x4);
            }
        }
    }
}

void World::Impl::updateBuffers(ScopedScratch scopeAlloc)
{
    const auto &scene = currentScene();

    {
        Array<Scene::RTInstance> rtInstances{scopeAlloc, scene.rtInstanceCount};
        Array<ModelInstance::Transforms> transforms{
            scopeAlloc, scene.modelInstances.size()};

        // The RTInstances generated here have to match the indices that get
        // assigned to tlas instances
        for (auto mi = 0u; mi < scene.modelInstances.size(); ++mi)
        {
            const auto &instance = scene.modelInstances[mi];
            transforms.push_back(instance.transforms);
            for (const auto &model : _data._models[instance.modelID].subModels)
            {
                rtInstances.push_back(Scene::RTInstance{
                    .modelInstanceID = mi,
                    .meshID = model.meshID,
                    .materialID = model.materialID,
                });
            }
        }

        // This is valid to offset (0) even on the first frame and we'll skip
        // reads anyway
        _byteOffsets.previousModelInstanceTransforms =
            _byteOffsets.modelInstanceTransforms;
        _byteOffsets.modelInstanceTransforms =
            _data._modelInstanceTransformsRing->write_elements(transforms);

        memcpy(
            scene.rtInstancesBuffer.mapped, rtInstances.data(),
            sizeof(Scene::RTInstance) * rtInstances.size());
    }

    updateTlasInstances(scopeAlloc.child_scope(), scene);

    _byteOffsets.directionalLight =
        scene.lights.directionalLight.write(*_lightDataRing);
    _byteOffsets.pointLights = scene.lights.pointLights.write(*_lightDataRing);
    _byteOffsets.spotLights = scene.lights.spotLights.write(*_lightDataRing);
}

bool World::Impl::buildAccelerationStructures(vk::CommandBuffer cb)
{
    if (_framesSinceFinalBlasBuilds > MAX_FRAMES_IN_FLIGHT)
    {
        // Be conservative and log this after we know the work is done. Let's
        // not worry about getting a tight time since this will only be off by
        // frametime at most.
        printf(
            "Streamed BLAS builds took %.2fs\n", _blasBuildTimer.getSeconds());
        _framesSinceFinalBlasBuilds = 0;
    }
    else if (_framesSinceFinalBlasBuilds > 0)
        _framesSinceFinalBlasBuilds++;

    bool blasAdded = false;
    WHEELS_ASSERT(_data._geometryMetadatas.size() == _data._meshInfos.size());
    if (_data._geometryMetadatas.size() > _data._blases.size())
    {
        const size_t maxBlasBuildsPerFrame = 10;
        const size_t unbuiltBlasCount =
            _data._geometryMetadatas.size() - _data._blases.size();
        const size_t blasBuildCount =
            std::min(unbuiltBlasCount, maxBlasBuildsPerFrame);
        for (size_t i = 0; i < blasBuildCount; ++i)
            buildNextBlas(cb);
        if (blasBuildCount == unbuiltBlasCount)
            _framesSinceFinalBlasBuilds = 1;
        blasAdded = true;
    }

    buildCurrentTlas(cb);

    return blasAdded;
}

AccelerationStructure World::Impl::createTlas(
    const Scene &scene, vk::AccelerationStructureBuildSizesInfoKHR sizeInfo,
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo)
{
    AccelerationStructure tlas;
    tlas.buffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeInfo.accelerationStructureSize,
                .usage =
                    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .debugName = "TLASBuffer",
    });

    const vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = tlas.buffer.handle,
        .size = sizeInfo.accelerationStructureSize,
        .type = buildInfo.type,
    };
    tlas.handle = _device->logical().createAccelerationStructureKHR(createInfo);
    tlas.address = _device->logical().getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR{
            .accelerationStructure = tlas.handle,
        });

    const vk::DescriptorBufferInfo instanceInfo{
        .buffer = scene.rtInstancesBuffer.handle, .range = VK_WHOLE_SIZE};

    StaticArray descriptorWrites{{
        vk::WriteDescriptorSet{
            .dstSet = scene.rtDescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        },
        vk::WriteDescriptorSet{
            .dstSet = scene.rtDescriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &instanceInfo,
        },
    }};

    // TODO:
    // This seems potentially messy to support with the
    // common reflection interface
    const vk::WriteDescriptorSetAccelerationStructureKHR asWrite{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas.handle,
    };
    descriptorWrites[0].pNext = &asWrite;

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);

    return tlas;
}

void World::Impl::drawSkybox(vk::CommandBuffer cb) const
{
    const vk::DeviceSize offset = 0;
    cb.bindVertexBuffers(
        0, 1, &_data._skyboxResources.vertexBuffer.handle, &offset);
    cb.draw(asserted_cast<uint32_t>(WorldData::sSkyboxVertsCount), 1, 0, 0);
}

void World::Impl::buildNextBlas(vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_data._geometryMetadatas.size() > _data._blases.size());

    const size_t targetMesh = _data._blases.size();
    if (targetMesh == 0)
        // TODO: This will continue to reset until the first blas is built.
        // Reset at the start of the first frame instead? Same for the material
        // timer?
        _blasBuildTimer.reset();

    const GeometryMetadata &metadata = _data._geometryMetadatas[targetMesh];
    if (metadata.bufferIndex == 0xFFFFFFFF)
        // Mesh is hasn't been uploaded yet
        return;

    _data._blases.push_back(AccelerationStructure{});
    auto &blas = _data._blases.back();

    const MeshInfo &info = _data._meshInfos[targetMesh];

    // Basics from RT Gems II chapter 16

    const Buffer &dataBuffer = _data._geometryBuffers[metadata.bufferIndex];
    WHEELS_ASSERT(dataBuffer.deviceAddress != 0);

    const vk::DeviceSize positionsOffset =
        metadata.positionsOffset * sizeof(uint32_t);
    const vk::DeviceSize indicesOffset =
        metadata.indicesOffset *
        (metadata.usesShortIndices == 1 ? sizeof(uint16_t) : sizeof(uint32_t));

    const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
        .vertexFormat = vk::Format::eR32G32B32Sfloat,
        .vertexData = dataBuffer.deviceAddress + positionsOffset,
        .vertexStride = 3 * sizeof(float),
        .maxVertex = info.vertexCount,
        .indexType = metadata.usesShortIndices == 1u ? vk::IndexType::eUint16
                                                     : vk::IndexType::eUint32,
        .indexData = dataBuffer.deviceAddress + indicesOffset,
    };

    const auto &material = _data._materials[info.materialID];
    const vk::GeometryFlagsKHR geomFlags =
        material.alphaMode == Material::AlphaMode::Opaque
            ? vk::GeometryFlagBitsKHR::eOpaque
            : vk::GeometryFlagsKHR{};
    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry = triangles,
        .flags = geomFlags,
    };
    const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
        .primitiveCount = info.indexCount / 3,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    // dst and scratch will be set once allocated
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geometry,
    };

    // TODO: This stuff is ~the same for TLAS and BLAS
    const auto sizeInfo =
        _device->logical().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
            {rangeInfo.primitiveCount});

    blas.buffer = _device->createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeInfo.accelerationStructureSize,
                .usage =
                    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eStorageBuffer,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .debugName = "BLASBuffer",
    });

    const vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = blas.buffer.handle,
        .size = sizeInfo.accelerationStructureSize,
        .type = buildInfo.type,
    };
    blas.handle = _device->logical().createAccelerationStructureKHR(createInfo);
    blas.address = _device->logical().getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR{
            .accelerationStructure = blas.handle,
        });

    buildInfo.dstAccelerationStructure = blas.handle;

    Buffer &scratchBuffer = reserveScratch(sizeInfo.buildScratchSize);
    WHEELS_ASSERT(scratchBuffer.deviceAddress != 0);

    buildInfo.scratchData = scratchBuffer.deviceAddress;

    // Make sure we can use the scratch
    scratchBuffer.transition(cb, BufferState::AccelerationStructureBuild);

    const auto *pRangeInfo = &rangeInfo;
    // TODO: Build multiple blas at a time
    cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

    // Make sure the following TLAS build waits until the BLAS is ready
    // TODO: Batch these barriers right before the tlas build
    blas.buffer.transition(cb, BufferState::AccelerationStructureBuild);
}

void World::Impl::buildCurrentTlas(vk::CommandBuffer cb)
{
    const Scene &scene = _data._scenes[_data._currentScene];
    AccelerationStructure &tlas = _data._tlases[_data._currentScene];

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
    vk::AccelerationStructureGeometryKHR geometry;
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo;
    createTlasBuildInfos(scene, rangeInfo, geometry, buildInfo, sizeInfo);

    // Let's not complicate things by duplicating the tlas build info and
    // instance update logic during load time. Should be fast enough to just do
    // this on the first frame that uses a given TLAS.
    if (!tlas.handle)
        tlas = createTlas(scene, sizeInfo, buildInfo);
    WHEELS_ASSERT(tlas.buffer.byteSize >= sizeInfo.accelerationStructureSize);

    buildInfo.dstAccelerationStructure = tlas.handle;

    Buffer &scratchBuffer = reserveScratch(sizeInfo.buildScratchSize);
    WHEELS_ASSERT(scratchBuffer.deviceAddress != 0);

    buildInfo.scratchData = scratchBuffer.deviceAddress;

    const vk::BufferCopy copyRegion{
        .srcOffset = _tlasInstancesUploadOffset,
        .dstOffset = 0,
        .size = _tlasInstancesBuffer.byteSize,
    };
    cb.copyBuffer(
        _tlasInstancesUploadRing->buffer(), _tlasInstancesBuffer.handle, 1,
        &copyRegion);

    const StaticArray barriers{{
        *scratchBuffer.transitionBarrier(
            BufferState::AccelerationStructureBuild, true),
        *tlas.buffer.transitionBarrier(
            BufferState::AccelerationStructureBuild, true),
    }};

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
    });

    const vk::AccelerationStructureBuildRangeInfoKHR *pRangeInfo = &rangeInfo;
    cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

    // First use needs to 'transition' the backing buffer into
    // RayTracingAccelerationStructureRead
}

Buffer &World::Impl::reserveScratch(vk::DeviceSize byteSize)
{
    // See if we have a big enough buffer available
    for (ScratchBuffer &sb : _scratchBuffers)
    {
        // Don't check for use within this frame as we assume barriers will be
        // used on the scratch buffer before use
        if (sb.buffer.byteSize >= byteSize)
        {
            sb.framesSinceLastUsed = 0;
            return sb.buffer;
        }
    }

    // Didn't find a viable buffer so allocate a new one
    _scratchBuffers.push_back(ScratchBuffer{
        .buffer = _device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = byteSize,
                    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eStorageBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .cacheDeviceAddress = true,
            .debugName = "ScratchBuffer",
        }),
    });

    return _scratchBuffers.back().buffer;
}

void World::Impl::reserveTlasInstances(uint32_t instanceCount)
{
    const vk::DeviceSize byteSize =
        sizeof(vk::AccelerationStructureInstanceKHR) * instanceCount;
    if (_tlasInstancesBuffer.byteSize < byteSize)
    {
        // TODO: This destroy isn't safe until all frames in flight have
        // finished
        _device->destroy(_tlasInstancesBuffer);
        // TODO: This destroy isn't safe until all frames in flight have
        // finished
        _tlasInstancesUploadRing.reset();

        _tlasInstancesBuffer = _device->createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = byteSize,
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::
                                 eAccelerationStructureBuildInputReadOnlyKHR,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .cacheDeviceAddress = true,
            .debugName = "InstancesBuffer",
        });

        const uint32_t ringByteSize = asserted_cast<uint32_t>(
            (byteSize + RingBuffer::sAlignment) * MAX_FRAMES_IN_FLIGHT);
        _tlasInstancesUploadRing = std::make_unique<RingBuffer>(
            _device, vk::BufferUsageFlagBits::eTransferSrc, ringByteSize,
            "InstancesUploadBuffer");
        _tlasInstancesUploadRing->startFrame();
    }
}

void World::Impl::updateTlasInstances(
    ScopedScratch scopeAlloc, const Scene &scene)
{
    // TODO:
    // Is it faster to poke instances directly into a mapped buffer instead
    // of collecting first and then passing them in one blob as initial
    // data?
    // Need to be careful to not cause read ops by accident, probably still use
    // memcpy for the write into the buffer.
    Array<vk::AccelerationStructureInstanceKHR> instances{
        scopeAlloc, scene.rtInstanceCount};
    uint32_t rti = 0;
    for (const auto &mi : scene.modelInstances)
    {
        const auto &model = _data._models[mi.modelID];

        // This has to be mat3x4 because we assume the transform already has
        // the same memory layout as vk::TransformationMatrixKHR
        const mat3x4 *trfn = &mi.transforms.modelToWorld;
        const vk::TransformMatrixKHR *trfn_cast =
            reinterpret_cast<const vk::TransformMatrixKHR *>(trfn);

        for (const auto &sm : model.subModels)
        {
            // Zero as accelerationStructureReference marks an inactive instance
            // according to the vk spec
            uint64_t asReference = 0;
            if (_data._blases.size() > sm.meshID)
            {
                const auto &blas = _data._blases[sm.meshID];
                asReference = blas.address;
            }

            instances.push_back(vk::AccelerationStructureInstanceKHR{
                .transform = *trfn_cast,
                .instanceCustomIndex = rti++,
                .mask = 0xFF,
                .accelerationStructureReference = asReference,
            });
        }
    }
    WHEELS_ASSERT(instances.size() == scene.rtInstanceCount);

    reserveTlasInstances(scene.rtInstanceCount);

    _tlasInstancesUploadOffset =
        _tlasInstancesUploadRing->write_elements(instances);
}

void World::Impl::createTlasBuildInfos(
    const Scene &scene,
    vk::AccelerationStructureBuildRangeInfoKHR &rangeInfoOut,
    vk::AccelerationStructureGeometryKHR &geometryOut,
    vk::AccelerationStructureBuildGeometryInfoKHR &buildInfoOut,
    vk::AccelerationStructureBuildSizesInfoKHR &sizeInfoOut)
{
    rangeInfoOut = vk::AccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = scene.rtInstanceCount,
        .primitiveOffset = 0,
    };

    WHEELS_ASSERT(_tlasInstancesBuffer.deviceAddress != 0);
    geometryOut = vk::AccelerationStructureGeometryKHR{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry =
            vk::AccelerationStructureGeometryInstancesDataKHR{
                .data = _tlasInstancesBuffer.deviceAddress,
            },
    };

    buildInfoOut = vk::AccelerationStructureBuildGeometryInfoKHR{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geometryOut,
    };

    sizeInfoOut = _device->logical().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfoOut,
        {rangeInfoOut.primitiveCount});
}

World::World(
    Allocator &generalAlloc, ScopedScratch scopeAlloc, Device *device,
    RingBuffer *constantsRing, const std::filesystem::path &scene)
: _impl{std::make_unique<World::Impl>(
      generalAlloc, WHEELS_MOV(scopeAlloc), device, constantsRing, scene)}
{
}

// Define here to have ~Impl defined
World::~World() = default;

void World::startFrame() { _impl->startFrame(); }

void World::endFrame() { _impl->endFrame(); }

bool World::handleDeferredLoading(vk::CommandBuffer cb, Profiler &profiler)
{
    return _impl->_data.handleDeferredLoading(cb, profiler);
}

void World::drawDeferredLoadingUi() const
{
    return _impl->_data.drawDeferredLoadingUi();
}

bool World::drawSceneUi() { return _impl->drawSceneUi(); }

bool World::drawCameraUi() { return _impl->drawCameraUi(); }

Scene &World::currentScene() { return _impl->currentScene(); }

const Scene &World::currentScene() const { return _impl->currentScene(); }

AccelerationStructure &World::currentTLAS() { return _impl->currentTLAS(); }

CameraParameters const &World::currentCamera() const
{
    WHEELS_ASSERT(_impl->_currentCamera < _impl->_data._cameras.size());
    return _impl->_data._cameras[_impl->_currentCamera];
}

bool World::isCurrentCameraDynamic() const
{
    WHEELS_ASSERT(_impl->_currentCamera < _impl->_data._cameraDynamic.size());
    return _impl->_data._cameraDynamic[_impl->_currentCamera];
}

void World::uploadMeshDatas(
    wheels::ScopedScratch scopeAlloc, uint32_t nextFrame)
{
    _impl->uploadMeshDatas(WHEELS_MOV(scopeAlloc), nextFrame);
}

void World::uploadMaterialDatas(uint32_t nextFrame, float lodBias)
{
    _impl->uploadMaterialDatas(nextFrame, lodBias);
}

void World::updateAnimations(float timeS, Profiler *profiler)
{
    _impl->updateAnimations(timeS, profiler);
}

void World::updateScene(
    ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
    Profiler *profiler)
{
    _impl->updateScene(WHEELS_MOV(scopeAlloc), cameraTransform, profiler);
}

void World::updateBuffers(ScopedScratch scopeAlloc)
{
    _impl->updateBuffers(WHEELS_MOV(scopeAlloc));
}

bool World::buildAccelerationStructures(vk::CommandBuffer cb)
{
    return _impl->buildAccelerationStructures(cb);
}

void World::drawSkybox(vk::CommandBuffer cb) const { _impl->drawSkybox(cb); }

const WorldDSLayouts &World::dsLayouts() const
{
    return _impl->_data._dsLayouts;
}

const WorldDescriptorSets &World::descriptorSets() const
{
    return _impl->_data._descriptorSets;
}

const WorldByteOffsets &World::byteOffsets() const
{
    return _impl->_byteOffsets;
}

Span<const Model> World::models() const { return _impl->_data._models; }

Span<const Material> World::materials() const
{
    return _impl->_data._materials;
}

Span<const MeshInfo> World::meshInfos() const
{
    return _impl->_data._meshInfos;
}

SkyboxResources &World::skyboxResources()
{
    return _impl->_data._skyboxResources;
}

size_t World::deferredLoadingGeneralAllocatorHighWatermark() const
{
    return _impl->_data._deferredLoadingGeneralAllocatorHighWatermark;
}

size_t World::linearAllocatorHighWatermark() const
{
    return _impl->_data.linearAllocatorHighWatermark();
}
