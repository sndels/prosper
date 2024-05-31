#include "World.hpp"

#include <glm/gtc/matrix_access.hpp>
#include <imgui.h>
#include <wheels/allocators/utils.hpp>
#include <wheels/containers/hash_set.hpp>

#include "../gfx/Device.hpp"
#include "../gfx/RingBuffer.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/SceneStats.hpp"
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool relativeEq(float a, float b, float maxRelativeDiff)
{
    const float diff = std::abs(a - b);
    const float maxMagnitude = std::max(std::abs(a), std::abs(b));
    const float scaledEpsilon = maxRelativeDiff * maxMagnitude;
    return diff < scaledEpsilon;
}

} // namespace

// TODO: Split scene loading and runtime scene into separate classes, CUs
class World::Impl
{
  public:
    Impl() noexcept = default;
    ~Impl();

    Impl(const Impl &other) = delete;
    Impl(Impl &&other) = delete;
    Impl &operator=(const Impl &other) = delete;
    Impl &operator=(Impl &&other) = delete;

    void init(
        ScopedScratch scopeAlloc, Device *device, RingBuffer *constantsRing,
        const std::filesystem::path &scene);

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
        SceneStats *sceneStats, Profiler *profiler);
    void updateBuffers(ScopedScratch scopeAlloc);
    // Has to be called after updateBuffers(). Returns true if new BLASes were
    // added.
    bool buildAccelerationStructures(
        ScopedScratch scopeAlloc, vk::CommandBuffer cb);
    void drawSkybox(vk::CommandBuffer cb) const;

  private:
    Buffer &reserveScratch(vk::DeviceSize byteSize);
    // Returns true if a blas build was queued
    bool buildNextBlas(ScopedScratch scopeAlloc, vk::CommandBuffer cb);
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

    RingBuffer *_constantsRing{nullptr};
    Device *_device{nullptr};
    RingBuffer _lightDataRing;
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
    Array<ScratchBuffer> _scratchBuffers{gAllocators.general};
    Buffer _tlasInstancesBuffer;
    OwningPtr<RingBuffer> _tlasInstancesUploadRing;
    uint32_t _tlasInstancesUploadOffset{0};
};

World::Impl::~Impl()
{
    if (_device != nullptr)
    {
        for (auto &sb : _scratchBuffers)
            _device->destroy(sb.buffer);
        _device->destroy(_tlasInstancesBuffer);
    }
}

void World::Impl::init(
    ScopedScratch scopeAlloc, Device *device, RingBuffer *constantsRing,
    const std::filesystem::path &scene)
{
    WHEELS_ASSERT(_device == nullptr);
    WHEELS_ASSERT(device != nullptr);
    WHEELS_ASSERT(constantsRing != nullptr);

    _device = device;
    _constantsRing = constantsRing;

    const uint32_t lightDataBufferSize =
        (DirectionalLight::sBufferByteSize + RingBuffer::sAlignment +
         PointLights::sBufferByteSize + RingBuffer::sAlignment +
         SpotLights::sBufferByteSize + RingBuffer::sAlignment) *
        MAX_FRAMES_IN_FLIGHT;
    _lightDataRing.init(
        _device, vk::BufferUsageFlagBits::eStorageBuffer, lightDataBufferSize,
        "LightDataRing");

    _data.init(
        WHEELS_MOV(scopeAlloc), device,
        WorldData::RingBuffers{
            .constantsRing = _constantsRing,
            .lightDataRing = &_lightDataRing,
        },
        scene);

    // This creates the instance ring and startFrame() assumes it exists
    reserveTlasInstances(1);
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
            scene.previousTransformsValid = false;
        }

        _data._currentScene = _nextScene.take();
    }
    _data._modelInstanceTransformsRing.startFrame();
    _lightDataRing.startFrame();
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
    scene.previousTransformsValid = true;
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
    SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);
    WHEELS_ASSERT(cameraTransform != nullptr);
    WHEELS_ASSERT(sceneStats != nullptr);

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

                sceneStats->totalNodeCount++;
                if (node.dynamicTransform)
                    sceneStats->animatedNodeCount++;
            }
        }
    }
}

void World::Impl::updateBuffers(ScopedScratch scopeAlloc)
{
    const auto &scene = currentScene();

    {
        Array<Scene::DrawInstance> drawInstances{
            scopeAlloc, scene.drawInstanceCount};
        Array<ModelInstance::Transforms> transforms{
            scopeAlloc, scene.modelInstances.size()};
        Array<float> scales{scopeAlloc, scene.modelInstances.size()};

        // The DrawInstances generated here have to match the indices that get
        // assigned to tlas instances
        for (auto mi = 0u; mi < scene.modelInstances.size(); ++mi)
        {
            const auto &instance = scene.modelInstances[mi];
            transforms.push_back(instance.transforms);

            const mat3x4 &modelToWorld = instance.transforms.modelToWorld;
            // lengths of rows instead of columns because of the transposed 3x4
            const vec3 scale{
                length(row(modelToWorld, 0)), length(row(modelToWorld, 1)),
                length(row(modelToWorld, 2))};

            // Zero scale indicates that the scale is non-uniform
            float uniformScale = 0.f;
            // 0.1mm precision should be plenty
            const float tolerance = 0.0001f;
            if (relativeEq(scale.x, scale.y, tolerance) &&
                relativeEq(scale.x, scale.z, tolerance))
                uniformScale = scale.x;
            scales.push_back(uniformScale);

            // Submodels are pushed one after another and TLAS instance update
            // assumes this as it uses the flattened index of the first submodel
            // as the custom index for each instance. RT shaders then access
            // each submodel from that using the geometry index of the hit.
            for (const auto &model : _data._models[instance.modelID].subModels)
            {
                drawInstances.push_back(Scene::DrawInstance{
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
            _data._modelInstanceTransformsRing.write_elements(transforms);
        _byteOffsets.modelInstanceScales =
            _data._modelInstanceTransformsRing.write_elements(scales);

        memcpy(
            scene.drawInstancesBuffer.mapped, drawInstances.data(),
            sizeof(Scene::DrawInstance) * drawInstances.size());
    }

    updateTlasInstances(scopeAlloc.child_scope(), scene);

    _byteOffsets.directionalLight =
        scene.lights.directionalLight.write(_lightDataRing);
    _byteOffsets.pointLights = scene.lights.pointLights.write(_lightDataRing);
    _byteOffsets.spotLights = scene.lights.spotLights.write(_lightDataRing);
}

bool World::Impl::buildAccelerationStructures(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb)
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
    if (_data._models.size() > _data._blases.size())
    {
        const size_t maxBlasBuildsPerFrame = 10;
        const size_t unbuiltBlasCount =
            _data._models.size() - _data._blases.size();
        const size_t blasBuildCount =
            std::min(unbuiltBlasCount, maxBlasBuildsPerFrame);
        size_t blasesBuilt = 0;
        for (; blasesBuilt < blasBuildCount; ++blasesBuilt)
        {
            if (!buildNextBlas(scopeAlloc.child_scope(), cb))
                break;
        }
        if (blasesBuilt == blasBuildCount && blasBuildCount == unbuiltBlasCount)
            _framesSinceFinalBlasBuilds = 1;
        blasAdded = blasesBuilt > 0;
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
        .buffer = scene.drawInstancesBuffer.handle, .range = VK_WHOLE_SIZE};

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

bool World::Impl::buildNextBlas(ScopedScratch scopeAlloc, vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_data._models.size() > _data._blases.size());

    const size_t modelIndex = _data._blases.size();
    if (modelIndex == 0)
        // TODO: This will continue to reset until the first blas is built.
        // Reset at the start of the first frame instead? Same for the material
        // timer?
        _blasBuildTimer.reset();

    const Model &model = _data._models[modelIndex];
    // Quick search through the submodels so we can early out if some of them
    // are not loaded in yet
    for (const Model::SubModel &sm : model.subModels)
    {
        const GeometryMetadata &metadata = _data._geometryMetadatas[sm.meshID];
        if (metadata.bufferIndex == 0xFFFFFFFF)
            // Mesh is hasn't been uploaded yet
            return false;
    }

    // Basics from RT Gems II chapter 16

    Array<vk::AccelerationStructureGeometryKHR> geometries{scopeAlloc};
    Array<vk::AccelerationStructureBuildRangeInfoKHR> rangeInfos{scopeAlloc};
    // vkGetAccelerationStructureBuildSizesKHR takes in just primitive counts
    // instead of the full range infos and there is no associated stride
    Array<uint32_t> maxPrimitiveCounts{scopeAlloc};
    geometries.reserve(model.subModels.size());
    rangeInfos.reserve(model.subModels.size());
    maxPrimitiveCounts.reserve(model.subModels.size());
    for (const Model::SubModel &sm : model.subModels)
    {
        const GeometryMetadata &metadata = _data._geometryMetadatas[sm.meshID];
        const MeshInfo &info = _data._meshInfos[sm.meshID];

        const Buffer &dataBuffer = _data._geometryBuffers[metadata.bufferIndex];
        WHEELS_ASSERT(dataBuffer.deviceAddress != 0);

        const vk::DeviceSize positionsOffset =
            metadata.positionsOffset * sizeof(uint32_t);
        const vk::DeviceSize indicesOffset =
            metadata.indicesOffset * (metadata.usesShortIndices == 1
                                          ? sizeof(uint16_t)
                                          : sizeof(uint32_t));

        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = dataBuffer.deviceAddress + positionsOffset,
            .vertexStride = 3 * sizeof(float),
            .maxVertex = info.vertexCount,
            .indexType = metadata.usesShortIndices == 1u
                             ? vk::IndexType::eUint16
                             : vk::IndexType::eUint32,
            .indexData = dataBuffer.deviceAddress + indicesOffset,
        };

        const Material &material = _data._materials[info.materialID];
        const vk::GeometryFlagsKHR geomFlags =
            material.alphaMode == Material::AlphaMode::Opaque
                ? vk::GeometryFlagBitsKHR::eOpaque
                : vk::GeometryFlagsKHR{};
        geometries.push_back(vk::AccelerationStructureGeometryKHR{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = triangles,
            .flags = geomFlags,
        });
        rangeInfos.push_back(vk::AccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = info.indexCount / 3,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        });
        maxPrimitiveCounts.push_back(rangeInfos.back().primitiveCount);
    }

    // dst and scratch will be set once allocated
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = asserted_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data(),
    };

    const vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
        _device->logical().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
            {asserted_cast<uint32_t>(maxPrimitiveCounts.size()),
             maxPrimitiveCounts.data()});

    _data._blases.push_back(AccelerationStructure{});
    AccelerationStructure &blas = _data._blases.back();

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

    // Let's just concatenate all the mesh names for the full debug name
    String blasName{scopeAlloc};
    for (const Model::SubModel &sm : model.subModels)
    {
        const String &smName = _data._meshNames[sm.meshID];
        blasName.extend(smName);
        blasName.push_back('|');
    }
    _device->logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eAccelerationStructureKHR,
            .objectHandle = reinterpret_cast<uint64_t>(
                static_cast<VkAccelerationStructureKHR>(blas.handle)),
            .pObjectName = blasName.c_str(),
        });

    buildInfo.dstAccelerationStructure = blas.handle;

    Buffer &scratchBuffer = reserveScratch(sizeInfo.buildScratchSize);
    WHEELS_ASSERT(scratchBuffer.deviceAddress != 0);

    buildInfo.scratchData = scratchBuffer.deviceAddress;

    // Make sure we can use the scratch
    scratchBuffer.transition(cb, BufferState::AccelerationStructureBuild);

    const vk::AccelerationStructureBuildRangeInfoKHR *pRangeInfo =
        rangeInfos.data();
    // TODO: Build multiple blas at a time
    cb.buildAccelerationStructuresKHR(1, &buildInfo, &pRangeInfo);

    // Make sure the following TLAS build waits until the BLAS is ready
    // TODO: Batch these barriers right before the tlas build
    blas.buffer.transition(cb, BufferState::AccelerationStructureBuild);

    return true;
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
        _tlasInstancesUploadRing = OwningPtr<RingBuffer>(gAllocators.general);
        _tlasInstancesUploadRing->init(
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
        scopeAlloc, scene.modelInstances.size()};
    uint32_t rti = 0;
    for (const auto &mi : scene.modelInstances)
    {
        const auto &model = _data._models[mi.modelID];

        // This has to be mat3x4 because we assume the transform already has
        // the same memory layout as vk::TransformationMatrixKHR
        const mat3x4 *trfn = &mi.transforms.modelToWorld;
        const vk::TransformMatrixKHR *trfn_cast =
            reinterpret_cast<const vk::TransformMatrixKHR *>(trfn);

        // Zero as accelerationStructureReference marks an inactive instance
        // according to the vk spec
        uint64_t asReference = 0;
        if (_data._blases.size() > mi.modelID)
        {
            const auto &blas = _data._blases[mi.modelID];
            asReference = blas.address;
        }

        instances.push_back(vk::AccelerationStructureInstanceKHR{
            .transform = *trfn_cast,
            .instanceCustomIndex = rti,
            .mask = 0xFF,
            .accelerationStructureReference = asReference,
        });
        // Draw instances pack all submodels of a model instance tightly so
        // let's use the index of the first one as the TLAS instance index. RT
        // shaders can then access each submodel from that using the geometry
        // index of the hit.
        rti += asserted_cast<uint32_t>(model.subModels.size());
    }
    WHEELS_ASSERT(instances.size() == scene.modelInstances.size());

    reserveTlasInstances(asserted_cast<uint32_t>(scene.modelInstances.size()));

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
        .primitiveCount = asserted_cast<uint32_t>(scene.modelInstances.size()),
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

World::World() noexcept
: _impl{gAllocators.general}
{
}

// Define here to have ~Impl defined
World::~World() = default;

void World::init(
    wheels::ScopedScratch scopeAlloc, Device *device, RingBuffer *constantsRing,
    const std::filesystem::path &scene)
{
    WHEELS_ASSERT(!_initialized);
    _impl->init(WHEELS_MOV(scopeAlloc), device, constantsRing, scene);
    _initialized = true;
}

void World::startFrame()
{
    WHEELS_ASSERT(_initialized);
    _impl->startFrame();
}

void World::endFrame()
{
    WHEELS_ASSERT(_initialized);
    _impl->endFrame();
}

bool World::handleDeferredLoading(vk::CommandBuffer cb, Profiler &profiler)
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data.handleDeferredLoading(cb, profiler);
}

bool World::unbuiltBlases() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._blases.size() < _impl->_data._models.size();
}

void World::drawDeferredLoadingUi() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data.drawDeferredLoadingUi();
}

bool World::drawSceneUi()
{
    WHEELS_ASSERT(_initialized);
    return _impl->drawSceneUi();
}

bool World::drawCameraUi()
{
    WHEELS_ASSERT(_initialized);
    return _impl->drawCameraUi();
}

Scene &World::currentScene()
{
    WHEELS_ASSERT(_initialized);
    return _impl->currentScene();
}

const Scene &World::currentScene() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->currentScene();
}

AccelerationStructure &World::currentTLAS()
{
    WHEELS_ASSERT(_initialized);
    return _impl->currentTLAS();
}

CameraParameters const &World::currentCamera() const
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(_impl->_currentCamera < _impl->_data._cameras.size());
    return _impl->_data._cameras[_impl->_currentCamera];
}

bool World::isCurrentCameraDynamic() const
{
    WHEELS_ASSERT(_initialized);
    WHEELS_ASSERT(_impl->_currentCamera < _impl->_data._cameraDynamic.size());
    return _impl->_data._cameraDynamic[_impl->_currentCamera];
}

void World::uploadMeshDatas(
    wheels::ScopedScratch scopeAlloc, uint32_t nextFrame)
{
    WHEELS_ASSERT(_initialized);
    _impl->uploadMeshDatas(WHEELS_MOV(scopeAlloc), nextFrame);
}

void World::uploadMaterialDatas(uint32_t nextFrame, float lodBias)
{
    WHEELS_ASSERT(_initialized);
    _impl->uploadMaterialDatas(nextFrame, lodBias);
}

void World::updateAnimations(float timeS, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    _impl->updateAnimations(timeS, profiler);
}

void World::updateScene(
    ScopedScratch scopeAlloc, CameraTransform *cameraTransform,
    SceneStats *sceneStats, Profiler *profiler)
{
    WHEELS_ASSERT(_initialized);
    _impl->updateScene(
        WHEELS_MOV(scopeAlloc), cameraTransform, sceneStats, profiler);
}

void World::updateBuffers(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(_initialized);
    _impl->updateBuffers(WHEELS_MOV(scopeAlloc));
}

bool World::buildAccelerationStructures(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_initialized);
    return _impl->buildAccelerationStructures(WHEELS_MOV(scopeAlloc), cb);
}

void World::drawSkybox(vk::CommandBuffer cb) const
{
    WHEELS_ASSERT(_initialized);
    _impl->drawSkybox(cb);
}

const WorldDSLayouts &World::dsLayouts() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._dsLayouts;
}

const WorldDescriptorSets &World::descriptorSets() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._descriptorSets;
}

const WorldByteOffsets &World::byteOffsets() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_byteOffsets;
}

Span<const Model> World::models() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._models;
}

Span<const Material> World::materials() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._materials;
}

Span<const MeshInfo> World::meshInfos() const
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._meshInfos;
}

SkyboxResources &World::skyboxResources()
{
    WHEELS_ASSERT(_initialized);
    return _impl->_data._skyboxResources;
}
