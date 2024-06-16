#include "WorldData.hpp"

#include <glm/gtc/type_ptr.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif // GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include "../gfx/Device.hpp"
#include <cstdio>
#include <imgui.h>

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sMaterialDatasReflectionSet = 0;
constexpr uint32_t sMaterialTexturesReflectionSet = 1;
constexpr uint32_t sGeometryReflectionSet = 0;
constexpr uint32_t sSceneInstancesReflectionSet = 0;
constexpr uint32_t sLightsReflectionSet = 0;
constexpr uint32_t sSkyboxReflectionSet = 0;

// This should be plenty while not being a ridiculously heavy descriptor set
// Need to know the limit up front to create the ds layout
constexpr size_t sMaxGeometryBuffersCount = 100;

constexpr int s_gl_nearest = 0x2600;
constexpr int s_gl_linear = 0x2601;
constexpr int s_gl_nearest_mipmap_nearest = 0x2700;
constexpr int s_gl_linear_mipmap_nearest = 0x2701;
constexpr int s_gl_nearest_mipmap_linear = 0x2702;
constexpr int s_gl_linear_mimpap_linear = 0x2703;
constexpr int s_gl_clamp_to_edge = 0x812F;
constexpr int s_gl_mirrored_repeat = 0x8370;
constexpr int s_gl_repeat = 0x2901;

void *cgltf_alloc_func(void *user, cgltf_size size)
{
    Allocator *alloc = static_cast<Allocator *>(user);
    return alloc->allocate(size);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) lib interface
void cgltf_free_func(void *user, void *ptr)
{
    Allocator *alloc = static_cast<Allocator *>(user);
    return alloc->deallocate(ptr);
}

const StaticArray sCgltfResultStr{{
    "cgltf_result_success",
    "cgltf_result_data_too_short",
    "cgltf_result_unknown_format",
    "cgltf_result_invalid_json",
    "cgltf_result_invalid_gltf",
    "cgltf_result_invalid_options",
    "cgltf_result_file_not_found",
    "cgltf_result_io_error",
    "cgltf_result_out_of_memory",
    "cgltf_result_legacy_gltf",
}};
// TODO:
// StaticArray<const char*, cgltf_result_max_enum> ctor should also complain at
// compile time
static_assert(
    sCgltfResultStr.size() == cgltf_result_max_enum,
    "Missing cgltf_result strings");

const StaticArray sCgltfAlphaModeStr{{
    "cgltf_alpha_mode_opaque",
    "cgltf_alpha_mode_mask",
    "cgltf_alpha_mode_blend",
}};
static_assert(
    sCgltfAlphaModeStr.size() == cgltf_alpha_mode_max_enum,
    "Missing cgltf_alpha_mode strings");

const StaticArray sCgltfLightTypeStr{{
    "cgltf_light_type_invalid",
    "cgltf_light_type_directional",
    "cgltf_light_type_point",
    "cgltf_light_type_spot",
}};
static_assert(
    sCgltfLightTypeStr.size() == cgltf_light_type_max_enum,
    "Missing cgltf_light_type strings");

const StaticArray sCgltfCameraTypeStr{{
    "cgltf_camera_type_invalid",
    "cgltf_camera_type_perspective",
    "cgltf_camera_type_orthographic",
}};
static_assert(
    sCgltfCameraTypeStr.size() == cgltf_camera_type_max_enum,
    "Missing cgltf_camera_type strings");

cgltf_data *loadGltf(const std::filesystem::path &path)
{
    cgltf_file_type gltfType = cgltf_file_type_invalid;
    if (path.extension() == ".gltf")
        gltfType = cgltf_file_type_gltf;
    else if (path.extension() == ".glb")
        gltfType = cgltf_file_type_glb;
    else
        throw std::runtime_error(
            "Unknown extension '" + path.extension().string() + "'");

    static_assert(std::is_same_v<decltype(gAllocators.general), TlsfAllocator>);
    const cgltf_options options{
        .type = gltfType,
        .memory = cgltf_memory_options{
            .alloc_func = cgltf_alloc_func,
            .free_func = cgltf_free_func,
            .user_data = &gAllocators.general,
        }};

    cgltf_data *data = nullptr;
    cgltf_result result =
        cgltf_parse_file(&options, path.string().c_str(), &data);
    if (result != cgltf_result_success)
        throw std::runtime_error(
            "Failed to parse gltf '" + path.string() +
            "': " + sCgltfResultStr[result]);

    result = cgltf_load_buffers(&options, data, path.string().c_str());
    if (result != cgltf_result_success)
        throw std::runtime_error(
            std::string("Failed to load glTF buffers: ") +
            sCgltfResultStr[result]);

    return data;
}

Buffer createSkyboxVertexBuffer()
{
    // Avoid large global allocation
    const StaticArray<glm::vec3, WorldData::sSkyboxVertsCount> skyboxVerts{{
        vec3{-1.0f, 1.0f, -1.0f},  vec3{-1.0f, -1.0f, -1.0f},
        vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, -1.0f},
        vec3{1.0f, 1.0f, -1.0f},   vec3{-1.0f, 1.0f, -1.0f},

        vec3{-1.0f, -1.0f, 1.0f},  vec3{-1.0f, -1.0f, -1.0f},
        vec3{-1.0f, 1.0f, -1.0f},  vec3{-1.0f, 1.0f, -1.0f},
        vec3{-1.0f, 1.0f, 1.0f},   vec3{-1.0f, -1.0f, 1.0f},

        vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, 1.0f},
        vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
        vec3{1.0f, 1.0f, -1.0f},   vec3{1.0f, -1.0f, -1.0f},

        vec3{-1.0f, -1.0f, 1.0f},  vec3{-1.0f, 1.0f, 1.0f},
        vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
        vec3{1.0f, -1.0f, 1.0f},   vec3{-1.0f, -1.0f, 1.0f},

        vec3{-1.0f, 1.0f, -1.0f},  vec3{1.0f, 1.0f, -1.0f},
        vec3{1.0f, 1.0f, 1.0f},    vec3{1.0f, 1.0f, 1.0f},
        vec3{-1.0f, 1.0f, 1.0f},   vec3{-1.0f, 1.0f, -1.0f},

        vec3{-1.0f, -1.0f, -1.0f}, vec3{-1.0f, -1.0f, 1.0f},
        vec3{1.0f, -1.0f, -1.0f},  vec3{1.0f, -1.0f, -1.0f},
        vec3{-1.0f, -1.0f, 1.0f},  vec3{1.0f, -1.0f, 1.0f},
    }};

    return gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeof(skyboxVerts[0]) * skyboxVerts.size(),
                .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            },
        .initialData = skyboxVerts.data(),
        .debugName = "SkyboxVertexBuffer",
    });
}

constexpr vk::Filter getVkFilterMode(cgltf_int glEnum)
{
    switch (glEnum)
    {
    case s_gl_nearest:
    case s_gl_nearest_mipmap_nearest:
    case s_gl_nearest_mipmap_linear:
        return vk::Filter::eNearest;
    case s_gl_linear:
    case s_gl_linear_mipmap_nearest:
    case s_gl_linear_mimpap_linear:
        return vk::Filter::eLinear;
    }

    fprintf(stderr, "Invalid gl filter %d\n", glEnum);
    return vk::Filter::eLinear;
}

constexpr vk::SamplerAddressMode getVkAddressMode(cgltf_int glEnum)
{
    switch (glEnum)
    {
    case s_gl_clamp_to_edge:
        return vk::SamplerAddressMode::eClampToEdge;
    case s_gl_mirrored_repeat:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case s_gl_repeat:
        return vk::SamplerAddressMode::eRepeat;
    }
    fprintf(stderr, "Invalid gl wrapping mode %d\n", glEnum);
    return vk::SamplerAddressMode::eClampToEdge;
}

// Appends data used by accessor to rawData and returns the pointer to it
const void *appendAccessorData(
    Array<uint8_t> &rawData, const cgltf_accessor &accessor)
{
    WHEELS_ASSERT(accessor.buffer_view != nullptr);
    const cgltf_buffer_view &view = *accessor.buffer_view;
    const uint32_t offset =
        asserted_cast<uint32_t>(accessor.offset + view.offset);

    WHEELS_ASSERT(accessor.buffer_view->buffer != nullptr);
    const cgltf_buffer &buffer = *view.buffer;

    const void *ret = rawData.data() + rawData.size();

    WHEELS_ASSERT(offset < buffer.size && view.size <= buffer.size - offset);
    const uint8_t *source = static_cast<const uint8_t *>(buffer.data) + offset;
    rawData.extend(Span<const uint8_t>{source, view.size});

    return ret;
}

} // namespace

WorldData::~WorldData()
{
    // Don't check for _initialized as we might be cleaning up after a failed
    // init.

    // Make sure the deferred loader exits before we clean up any shared
    // resources.
    if (_deferredLoadingContext.has_value())
    {
        _deferredLoadingContext->kill();
        // Copy over any new geometry buffers as ~WorldData is responsible of
        // destroying them
        while (_deferredLoadingContext->geometryBuffers.size() >
               _geometryBuffers.size())
        {
            _geometryBuffers.push_back(
                _deferredLoadingContext
                    ->geometryBuffers[_geometryBuffers.size()]
                    .clone());
        }
    }

    gDevice.logical().destroy(_dsLayouts.lights);
    gDevice.logical().destroy(_dsLayouts.skybox);
    gDevice.logical().destroy(_dsLayouts.rayTracing);
    gDevice.logical().destroy(_dsLayouts.sceneInstances);
    gDevice.logical().destroy(_dsLayouts.geometry);
    gDevice.logical().destroy(_dsLayouts.materialTextures);
    gDevice.logical().destroy(_dsLayouts.materialDatas);

    gDevice.destroy(_skyboxResources.vertexBuffer);
    for (const vk::ImageView view : _skyboxResources.radianceViews)
        gDevice.logical().destroy(view);
    gDevice.destroy(_skyboxResources.radiance);
    gDevice.destroy(_skyboxResources.specularBrdfLut);
    gDevice.destroy(_skyboxResources.irradiance);
    gDevice.logical().destroy(_skyboxResources.sampler);

    for (Buffer &buffer : _materialsBuffers)
        gDevice.destroy(buffer);

    for (AccelerationStructure &blas : _blases)
    {
        gDevice.logical().destroy(blas.handle);
        gDevice.destroy(blas.buffer);
    }
    for (AccelerationStructure &tlas : _tlases)
    {
        gDevice.logical().destroy(tlas.handle);
        gDevice.destroy(tlas.buffer);
    }
    for (Scene &scene : _scenes)
        gDevice.destroy(scene.drawInstancesBuffer);
    for (Buffer &buffer : _geometryBuffers)
        gDevice.destroy(buffer);
    for (Buffer &buffer : _geometryMetadatasBuffers)
        gDevice.destroy(buffer);
    for (Buffer &buffer : _meshletCountsBuffers)
        gDevice.destroy(buffer);
    for (const vk::Sampler sampler : _samplers)
        gDevice.logical().destroy(sampler);
}

void WorldData::init(
    ScopedScratch scopeAlloc, const RingBuffers &ringBuffers,
    const std::filesystem::path &scene)
{
    WHEELS_ASSERT(!_initialized);

    _descriptorAllocator.init();
    _sceneDir = resPath(scene.parent_path());
    _skyboxResources.vertexBuffer = createSkyboxVertexBuffer();
    _skyboxResources.texture.init(
        scopeAlloc.child_scope(), resPath("env/storm.ktx"));

    _skyboxResources.irradiance = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = SkyboxResources::sSkyboxIrradianceResolution,
                .height = SkyboxResources::sSkyboxIrradianceResolution,
                .layerCount = 6,
                .createFlags = vk::ImageCreateFlagBits::eCubeCompatible,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
        .debugName = "SkyboxIrradiance",
    });
    {
        const vk::CommandBuffer cb = gDevice.beginGraphicsCommands();
        _skyboxResources.irradiance.transition(
            cb, ImageState::FragmentShaderSampledRead |
                    ImageState::ComputeShaderSampledRead |
                    ImageState::RayTracingSampledRead);
        gDevice.endGraphicsCommands(cb);
    }

    _skyboxResources.specularBrdfLut = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR16G16Unorm,
                .width = SkyboxResources::sSpecularBrdfLutResolution,
                .height = SkyboxResources::sSpecularBrdfLutResolution,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
        .debugName = "SpecularBrdfLut",
    });
    {
        const vk::CommandBuffer cb = gDevice.beginGraphicsCommands();
        _skyboxResources.specularBrdfLut.transition(
            cb, ImageState::FragmentShaderSampledRead |
                    ImageState::ComputeShaderSampledRead |
                    ImageState::RayTracingSampledRead);
        gDevice.endGraphicsCommands(cb);
    }

    const uint32_t radianceMips =
        asserted_cast<uint32_t>(floor(std::log2((
            static_cast<float>(SkyboxResources::sSkyboxRadianceResolution))))) +
        1;
    _skyboxResources.radiance = gDevice.createImage(ImageCreateInfo{
        .desc =
            ImageDescription{
                .format = vk::Format::eR16G16B16A16Sfloat,
                .width = SkyboxResources::sSkyboxRadianceResolution,
                .height = SkyboxResources::sSkyboxRadianceResolution,
                .mipCount = radianceMips,
                .layerCount = 6,
                .createFlags = vk::ImageCreateFlagBits::eCubeCompatible,
                .usageFlags = vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eStorage,
            },
        .debugName = "SkyboxRadiance",
    });
    for (uint32_t i = 0; i < radianceMips; ++i)
        _skyboxResources.radianceViews.push_back(
            gDevice.logical().createImageView(vk::ImageViewCreateInfo{
                .image = _skyboxResources.radiance.handle,
                .viewType = vk::ImageViewType::eCube,
                .format = _skyboxResources.radiance.format,
                .subresourceRange =
                    vk::ImageSubresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = i,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 6,
                    },
            }));
    gDevice.logical().setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT{
            .objectType = vk::ObjectType::eImageView,
            .objectHandle = reinterpret_cast<uint64_t>(static_cast<VkImageView>(
                _skyboxResources.radianceViews.back())),
            .pObjectName = "SkyboxRadiance",
        });
    {
        const vk::CommandBuffer cb = gDevice.beginGraphicsCommands();
        _skyboxResources.radiance.transition(
            cb, ImageState::FragmentShaderSampledRead |
                    ImageState::ComputeShaderSampledRead |
                    ImageState::RayTracingSampledRead);
        gDevice.endGraphicsCommands(cb);
    }

    _skyboxResources.sampler =
        gDevice.logical().createSampler(vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
        });

    printf("Loading world\n");

    const std::filesystem::path fullScenePath = resPath(scene);
    if (!std::filesystem::exists(fullScenePath))
        throw std::runtime_error(
            "Couldn't find '" + fullScenePath.string() + "'");

    const std::filesystem::file_time_type sourceWriteTime =
        std::filesystem::last_write_time(fullScenePath);

    Timer t;
    cgltf_data *gltfData = loadGltf(fullScenePath);
    WHEELS_ASSERT(gltfData != nullptr);
    printf("glTF model loading took %.2fs\n", t.getSeconds());

    _deferredLoadingContext.emplace();
    // Deferred context is responsible for freeing gltfData. Dispatch happens
    // after other loading finishes and WorldData will then always go through
    // the deferred context when it needs gltfData.
    _deferredLoadingContext->init(_sceneDir, sourceWriteTime, gltfData);

    const auto &tl = [&](const char *stage, std::function<void()> const &fn)
    {
        t.reset();
        fn();
        printf("%s took %.2fs\n", stage, t.getSeconds());
    };

    Array<Texture2DSampler> texture2DSamplers{
        gAllocators.general, gltfData->images_count + 1};
    tl("Texture loading",
       [&]() {
           loadTextures(scopeAlloc.child_scope(), *gltfData, texture2DSamplers);
       });
    tl("Material loading",
       [&]() { loadMaterials(*gltfData, texture2DSamplers); });
    tl("Model loading ",
       [&]() { loadModels(scopeAlloc.child_scope(), *gltfData); });
    tl("Animation and scene loading ",
       [&]()
       {
           const HashMap<uint32_t, NodeAnimations> nodeAnimations =
               loadAnimations(scopeAlloc.child_scope(), *gltfData);
           loadScenes(scopeAlloc.child_scope(), *gltfData, nodeAnimations);
       });
    tl("Buffer creation", [&]() { createBuffers(); });

    _tlases.resize(_scenes.size());

    reflectBindings(scopeAlloc.child_scope());
    createDescriptorSets(scopeAlloc.child_scope(), ringBuffers);

    _deferredLoadingContext->launch();
    _initialized = true;
}

void WorldData::uploadMeshDatas(ScopedScratch scopeAlloc, uint32_t nextFrame)
{
    if (!_deferredLoadingContext.has_value())
        return;

    if (_geometryGenerations[nextFrame] ==
        _deferredLoadingContext->geometryGeneration)
        return;

    {
        uint8_t *mapped =
            static_cast<uint8_t *>(_geometryMetadatasBuffers[nextFrame].mapped);
        memcpy(
            mapped, _geometryMetadatas.data(),
            _geometryMetadatas.size() * sizeof(_geometryMetadatas[0]));
    }
    {
        Array<uint32_t> meshletCounts{scopeAlloc, _meshInfos.size()};
        for (const MeshInfo &info : _meshInfos)
        {
            if (info.indexCount > 0)
                meshletCounts.push_back(info.meshletCount);
            else
                meshletCounts.push_back(0u);
        }
        uint32_t *mapped =
            static_cast<uint32_t *>(_meshletCountsBuffers[nextFrame].mapped);
        memcpy(
            mapped, meshletCounts.data(),
            meshletCounts.size() * sizeof(meshletCounts[0]));
    }

    _geometryGenerations[nextFrame] =
        _deferredLoadingContext->geometryGeneration;

    Array<vk::DescriptorBufferInfo> bufferInfos{
        scopeAlloc, 2 + _geometryBuffers.size()};

    bufferInfos.push_back(vk::DescriptorBufferInfo{
        .buffer = _geometryMetadatasBuffers[nextFrame].handle,
        .range = VK_WHOLE_SIZE,
    });
    bufferInfos.push_back(vk::DescriptorBufferInfo{
        .buffer = _meshletCountsBuffers[nextFrame].handle,
        .range = VK_WHOLE_SIZE,
    });

    WHEELS_ASSERT(
        _geometryBuffers.size() == _geometryBufferAllocatedByteCounts.size());
    const size_t bufferCount = _geometryBuffers.size();
    for (size_t i = 0; i < bufferCount; ++i)
    {
        if (_geometryBufferAllocatedByteCounts[i] == 0)
        {
            // We might push a new buffer before the mesh it got created for
            // gets copied over. Let's just skip in that case. Just make sure we
            // won't leave other already used buffers hanging.
            for (size_t j = i + 1; j < bufferCount; ++j)
                WHEELS_ASSERT(_geometryBufferAllocatedByteCounts[j] == 0);
            break;
        }

        bufferInfos.push_back(vk::DescriptorBufferInfo{
            .buffer = _geometryBuffers[i].handle,
            .range = _geometryBufferAllocatedByteCounts[i],
        });
    }

    const StaticArray descriptorInfos{{
        DescriptorInfo{bufferInfos[0]},
        DescriptorInfo{bufferInfos[1]},
        DescriptorInfo{bufferInfos.span(2, bufferInfos.size())},
    }};

    const Array descriptorWrites =
        _geometryReflection->generateDescriptorWrites(
            scopeAlloc, sGeometryReflectionSet,
            _descriptorSets.geometry[nextFrame], descriptorInfos);

    gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void WorldData::uploadMaterialDatas(uint32_t nextFrame)
{
    if (!_deferredLoadingContext.has_value())
        return;

    if (_materialsGenerations[nextFrame] ==
        _deferredLoadingContext->materialsGeneration)
        return;

    Material *mapped =
        static_cast<Material *>(_materialsBuffers[nextFrame].mapped);
    memcpy(
        mapped, _materials.data(), _materials.size() * sizeof(_materials[0]));

    _materialsGenerations[nextFrame] =
        _deferredLoadingContext->materialsGeneration;
}

bool WorldData::handleDeferredLoading(vk::CommandBuffer cb, Profiler &profiler)
{
    if (!_deferredLoadingContext.has_value())
        return false;

    DeferredLoadingContext &ctx = *_deferredLoadingContext;

    const bool allMeshesLoaded = ctx.loadedMeshCount == ctx.meshes.size();
    const bool allMaterialsLoaded =
        ctx.loadedMaterialCount == ctx.materials.size();

    if (allMeshesLoaded && allMaterialsLoaded)
    {
        WHEELS_ASSERT(
            ctx.loadedMeshCount == _meshInfos.size() &&
            "Meshes should have been loaded before textures");

        // Don't clean up until all in flight uploads are finished
        if (ctx.framesSinceFinish++ > MAX_FRAMES_IN_FLIGHT)
        {
            printf(
                "Material streaming took %.2fs\n",
                _materialStreamingTimer.getSeconds());

            _deferredLoadingContext.reset();
        }
        return false;
    }

    // No gpu as timestamps are flaky for this work
    const auto _s = profiler.createCpuScope("DeferredLoading");

    if (ctx.loadedImageCount == 0)
        _materialStreamingTimer.reset();

    bool newMeshAvailable = false;
    if (!allMeshesLoaded)
        newMeshAvailable = pollMeshWorker(cb);

    size_t newTexturesAvailable = 0;
    bool shouldUpdateMaterials = false;
    if (allMeshesLoaded)
    {
        // All materials loaded implies all images loaded
        WHEELS_ASSERT(!allMaterialsLoaded);

        if (ctx.loadedImageCount < ctx.gltfData->images_count)
        {
            newTexturesAvailable = pollTextureWorker(cb);
        }
        else
            // We should not get here if the model has any images
            WHEELS_ASSERT(
                ctx.gltfData->images_count == 0 &&
                ctx.loadedMaterialCount < ctx.gltfData->materials_count);
        shouldUpdateMaterials = true;
    }

    if (newTexturesAvailable > 0)
        updateDescriptorsWithNewTextures(newTexturesAvailable);

    const bool newMaterialsAvailable =
        shouldUpdateMaterials ? updateMaterials() : false;

    return newMeshAvailable || newMaterialsAvailable;
}

void WorldData::drawDeferredLoadingUi() const
{
    if (_deferredLoadingContext.has_value() || _blases.size() < _models.size())
    {
        ImGui::SetNextWindowPos(ImVec2{400, 50}, ImGuiCond_Appearing);
        ImGui::Begin(
            "DeferredLoadingProgress", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_AlwaysAutoResize);
        if (_deferredLoadingContext.has_value())
        {
            ImGui::Text(
                "Meshes loaded: %u/%u",
                _deferredLoadingContext->loadedMeshCount,
                asserted_cast<uint32_t>(_meshInfos.size()));
            ImGui::Text(
                "Images loaded: %u/%u",
                _deferredLoadingContext->loadedImageCount,
                asserted_cast<uint32_t>(
                    _deferredLoadingContext->gltfData->textures_count));
        }
        ImGui::End();
    }
}

void WorldData::loadTextures(
    ScopedScratch scopeAlloc, const cgltf_data &gltfData,
    Array<Texture2DSampler> &texture2DSamplers)
{
    {
        const vk::SamplerCreateInfo info{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear, // TODO
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        _samplers.push_back(gDevice.logical().createSampler(info));
    }
    WHEELS_ASSERT(
        gltfData.samplers_count < 0xFE &&
        "Too many samplers to pack in u32 texture index");
    for (const cgltf_sampler &sampler :
         Span{gltfData.samplers, gltfData.samplers_count})
    {
        const vk::SamplerCreateInfo info{
            .magFilter = getVkFilterMode(sampler.mag_filter),
            .minFilter = getVkFilterMode(sampler.min_filter),
            .mipmapMode = vk::SamplerMipmapMode::eLinear, // TODO
            .addressModeU = getVkAddressMode(sampler.wrap_s),
            .addressModeV = getVkAddressMode(sampler.wrap_t),
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = VK_TRUE, // TODO: Is there a gltf flag?
            .maxAnisotropy = 16,
            .minLod = 0,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        _samplers.push_back(gDevice.logical().createSampler(info));
    }

    Buffer stagingBuffer = createTextureStaging();

    _texture2Ds.reserve(gltfData.images_count + 1);
    {
        const vk::CommandBuffer cb = gDevice.beginGraphicsCommands();
        _texture2Ds.emplace_back();
        _texture2Ds.back().init(
            scopeAlloc.child_scope(), resPath("texture/empty.png"), cb,
            stagingBuffer, false,
            ImageState::FragmentShaderRead | ImageState::RayTracingRead);
        gDevice.endGraphicsCommands(cb);

        texture2DSamplers.emplace_back();
    }

    WHEELS_ASSERT(
        gltfData.images_count < 0xFFFFFE &&
        "Too many textures to pack in u32 texture index");

    gDevice.destroy(stagingBuffer);

    for (const cgltf_texture &texture :
         Span{gltfData.textures, gltfData.textures_count})
    {
        WHEELS_ASSERT(texture.image != nullptr);

        const uint32_t imageIndex = asserted_cast<uint32_t>(
            cgltf_image_index(&gltfData, texture.image) + 1);
        const uint32_t samplerIndex =
            texture.sampler == nullptr
                ? 0
                : asserted_cast<uint32_t>(
                      cgltf_sampler_index(&gltfData, texture.sampler) + 1);
        texture2DSamplers.emplace_back(imageIndex, samplerIndex);
    }
}

void WorldData::loadMaterials(
    const cgltf_data &gltfData,
    const Array<Texture2DSampler> &texture2DSamplers)
{
    _materials.push_back(Material{});

    for (const cgltf_material &material :
         Span{gltfData.materials, gltfData.materials_count})
    {
        Material mat;
        if (material.has_pbr_metallic_roughness == 0)
        {
            fprintf(
                stderr, "'%s' doesn't have pbr metallic roughness components\n",
                material.name);
            continue;
        }

        auto const getTexture2dSampler =
            [&](cgltf_texture_view const &tv,
                const char *channelName) -> Texture2DSampler
        {
            if (tv.texture != nullptr)
            {
                if (tv.has_transform == 1)
                    fprintf(
                        stderr, "%s: %s has a transform\n", material.name,
                        channelName);
                if (tv.scale != 1.f)
                    fprintf(
                        stderr, "%s: %s Scale isn't 1\n", material.name,
                        channelName);
                if (tv.texcoord != 0)
                    fprintf(
                        stderr, "%s: %s TexCoord isn't 0\n", material.name,
                        channelName);

                const cgltf_size index =
                    cgltf_texture_index(&gltfData, tv.texture);
                return texture2DSamplers[index + 1];
            }
            return Texture2DSampler{};
        };

        const cgltf_pbr_metallic_roughness &pbrParams =
            material.pbr_metallic_roughness;

        mat.baseColor =
            getTexture2dSampler(pbrParams.base_color_texture, "base color");
        mat.baseColorFactor = make_vec4(&pbrParams.base_color_factor[0]);
        mat.metallicRoughness = getTexture2dSampler(
            pbrParams.metallic_roughness_texture, "metallic roughness");
        mat.metallicFactor = pbrParams.metallic_factor;
        mat.roughnessFactor = pbrParams.roughness_factor;
        mat.normal = getTexture2dSampler(material.normal_texture, "normal");
        if (material.alpha_mode == cgltf_alpha_mode_mask)
            mat.alphaMode = Material::AlphaMode::Mask;
        else if (material.alpha_mode == cgltf_alpha_mode_blend)
            mat.alphaMode = Material::AlphaMode::Blend;
        else if (material.alpha_mode != cgltf_alpha_mode_opaque)
            fprintf(
                stderr, "%s: Unsupported alpha mode '%s'\n", material.name,
                sCgltfAlphaModeStr[material.alpha_mode]);
        mat.alphaCutoff = material.alpha_cutoff;

        // Copy the alpha mode of the real material because that's used to
        // set opaque flag in rt
        _materials.push_back(Material{
            .alphaMode = mat.alphaMode,
        });
        WHEELS_ASSERT(_deferredLoadingContext.has_value());
        _deferredLoadingContext->materials.push_back(mat);
    }
}

void WorldData::loadModels(ScopedScratch scopeAlloc, const cgltf_data &gltfData)
{
    _models.reserve(gltfData.meshes_count);

    size_t totalPrimitiveCount = 0;
    for (const cgltf_mesh &mesh : Span{gltfData.meshes, gltfData.meshes_count})
        totalPrimitiveCount += mesh.primitives_count;
    _geometryMetadatas.resize(totalPrimitiveCount);
    _meshInfos.resize(totalPrimitiveCount);

    uint32_t meshID = 0;
    for (cgltf_size mi = 0; mi < gltfData.meshes_count; ++mi)
    {
        const cgltf_mesh &mesh = gltfData.meshes[mi];

        _models.emplace_back(gAllocators.world);
        Model &model = _models.back();

        model.subModels.reserve(mesh.primitives_count);
        for (cgltf_size pi = 0; pi < mesh.primitives_count; ++pi)
        {
            const cgltf_primitive &primitive = mesh.primitives[pi];
            WHEELS_ASSERT(primitive.indices != nullptr);

            InputGeometryMetadata inputMetadata{
                .indices = primitive.indices,
                .sourceMeshIndex = asserted_cast<uint32_t>(mi),
                .sourcePrimitiveIndex = asserted_cast<uint32_t>(pi),
            };

            for (cgltf_size ai = 0; ai < primitive.attributes_count; ++ai)
            {
                const cgltf_attribute &attr = primitive.attributes[ai];
                WHEELS_ASSERT(attr.data != nullptr);

                if (strcmp("POSITION", attr.name) == 0)
                    inputMetadata.positions = attr.data;
                else if (strcmp("NORMAL", attr.name) == 0)
                    inputMetadata.normals = attr.data;
                else if (strcmp("TANGENT", attr.name) == 0)
                    inputMetadata.tangents = attr.data;
                else if (strcmp("TEXCOORD_0", attr.name) == 0)
                    inputMetadata.texCoord0s = attr.data;
            }
            WHEELS_ASSERT(inputMetadata.positions != nullptr);
            WHEELS_ASSERT(inputMetadata.normals != nullptr);
            WHEELS_ASSERT(
                inputMetadata.positions->count == inputMetadata.normals->count);
            WHEELS_ASSERT(
                inputMetadata.tangents == nullptr ||
                inputMetadata.tangents->count ==
                    inputMetadata.positions->count);
            WHEELS_ASSERT(
                inputMetadata.texCoord0s == nullptr ||
                inputMetadata.texCoord0s->count ==
                    inputMetadata.positions->count);

            const uint32_t material =
                primitive.material != nullptr
                    ? asserted_cast<uint32_t>(
                          cgltf_material_index(&gltfData, primitive.material) +
                          1)
                    : 0;

            const MeshInfo meshInfo = MeshInfo{
                .vertexCount =
                    asserted_cast<uint32_t>(inputMetadata.positions->count),
                .indexCount =
                    asserted_cast<uint32_t>(inputMetadata.indices->count),
                .materialID = material,
            };

            WHEELS_ASSERT(
                _deferredLoadingContext.has_value() &&
                !_deferredLoadingContext->worker.has_value() &&
                "Loading worker is running while input data is being set "
                "up");
            _deferredLoadingContext->meshes.emplace_back(
                inputMetadata, meshInfo);
            // Don't set metadata or info for the mesh index as default
            // values signal invalid or not yet loaded for other parts. Tangents
            // generation might also change the number of unique vertices.

            model.subModels.push_back(Model::SubModel{
                .meshID = meshID++,
                .materialID = material,
            });
        }
    }

    for (size_t i = 0; i < _geometryMetadatasBuffers.size(); ++i)
        _geometryMetadatasBuffers[i] = gDevice.createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = asserted_cast<uint32_t>(
                        _geometryMetadatas.size() * sizeof(GeometryMetadata)),
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .initialData = _geometryMetadatas.data(),
            .debugName = "GeometryMetadatas",
        });

    Array<uint32_t> zeroMeshletCounts{scopeAlloc};
    zeroMeshletCounts.resize(_meshInfos.size());
    memset(
        zeroMeshletCounts.data(), 0,
        zeroMeshletCounts.size() * sizeof(uint32_t));
    for (size_t i = 0; i < _meshletCountsBuffers.size(); ++i)
        _meshletCountsBuffers[i] = gDevice.createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = asserted_cast<uint32_t>(
                        _meshInfos.size() * sizeof(uint32_t)),
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .initialData = zeroMeshletCounts.data(),
            .debugName = "MeshletCounts",
        });
}

HashMap<uint32_t, WorldData::NodeAnimations> WorldData::loadAnimations(
    ScopedScratch scopeAlloc, const cgltf_data &gltfData)
{
    // First find out the amount of memory we need to copy over all the
    // animation data. We need to do that before parsing animations because
    // we'll store pointers to the raw data and it can't be resized during/after
    // the gather.
    // Also gather sizes for the animation arrays because we'll store pointers
    // to them in the map.
    uint32_t totalAnimationBytes = 0;
    uint32_t totalVec3Animations = 0;
    uint32_t totalQuatAnimations = 0;
    for (const cgltf_animation &animation :
         Span{gltfData.animations, gltfData.animations_count})
    {
        for (const cgltf_animation_sampler &sampler :
             Span{animation.samplers, animation.samplers_count})
        {
            {
                WHEELS_ASSERT(sampler.input != nullptr);
                const cgltf_accessor &accessor = *sampler.input;
                WHEELS_ASSERT(accessor.buffer_view != nullptr);
                const cgltf_buffer_view &view = *accessor.buffer_view;
                totalAnimationBytes += asserted_cast<uint32_t>(view.size);
            }
            {
                WHEELS_ASSERT(sampler.output != nullptr);
                const cgltf_accessor &accessor = *sampler.output;
                WHEELS_ASSERT(accessor.buffer_view != nullptr);
                const cgltf_buffer_view &view = *accessor.buffer_view;
                totalAnimationBytes += asserted_cast<uint32_t>(view.size);
                if (accessor.type == cgltf_type_vec3)
                    totalVec3Animations++;
                else if (accessor.type == cgltf_type_vec4)
                    // Only quaternion animations are currently sampled from
                    // vec4 outputs
                    totalQuatAnimations++;
            }
        }
    }

    // Init empty animations for all nodes and avoid resizes by safe upper bound
    // TODO: Why is this a map and not a vector?
    HashMap<uint32_t, NodeAnimations> ret{
        gAllocators.general, gltfData.nodes_count * 2};
    for (uint32_t i = 0; i < gltfData.nodes_count; ++i)
        ret.insert_or_assign(i, NodeAnimations{});

    // Now reserve the data so that our pointers are stable when we push the
    // data
    _rawAnimationData.reserve(totalAnimationBytes);
    _animations._vec3.reserve(totalVec3Animations);
    _animations._quat.reserve(totalQuatAnimations);
    for (const cgltf_animation &animation :
         Span{gltfData.animations, gltfData.animations_count})
    {
        // Map loaded animations to indices in gltf samplers
        // We know the types of these based on the target property so let's do
        // the dirty casts to void and back to store them
        Array<void *> concreteAnimations{scopeAlloc, animation.channels_count};
        for (cgltf_size si = 0; si < animation.samplers_count; ++si)
        {
            const cgltf_animation_sampler &sampler = animation.samplers[si];

            InterpolationType interpolation{InterpolationType::Step};
            if (sampler.interpolation == cgltf_interpolation_type_step)
                interpolation = InterpolationType::Step;
            else if (sampler.interpolation == cgltf_interpolation_type_linear)
                interpolation = InterpolationType::Linear;
            else if (
                sampler.interpolation == cgltf_interpolation_type_cubic_spline)
                interpolation = InterpolationType::CubicSpline;
            else
                WHEELS_ASSERT(!"Unsupported interpolation type");

            WHEELS_ASSERT(sampler.input != nullptr);
            const cgltf_accessor &inputAccessor = *sampler.input;
            WHEELS_ASSERT(!inputAccessor.is_sparse);
            WHEELS_ASSERT(
                inputAccessor.component_type == cgltf_component_type_r_32f);
            WHEELS_ASSERT(inputAccessor.type == cgltf_type_scalar);

            // TODO:
            // Share data for accessors that use the same bytes?
            const float *timesPtr = static_cast<const float *>(
                appendAccessorData(_rawAnimationData, inputAccessor));

            WHEELS_ASSERT(inputAccessor.has_min);
            WHEELS_ASSERT(inputAccessor.has_max);
            TimeAccessor timeFrames{
                timesPtr, asserted_cast<uint32_t>(inputAccessor.count),
                TimeAccessor::Interval{
                    .startTimeS = static_cast<float>(inputAccessor.min[0]),
                    .endTimeS = static_cast<float>(inputAccessor.max[0]),
                }};

            WHEELS_ASSERT(sampler.output != nullptr);
            const cgltf_accessor &outputAccessor = *sampler.output;
            WHEELS_ASSERT(!outputAccessor.is_sparse);
            WHEELS_ASSERT(
                outputAccessor.component_type == cgltf_component_type_r_32f);

            // TODO:
            // Share data for accessors that use the same bytes?
            const uint8_t *valuesPtr = static_cast<const uint8_t *>(
                appendAccessorData(_rawAnimationData, outputAccessor));

            if (outputAccessor.type == cgltf_type_vec3)
            {
                ValueAccessor<vec3> valueFrames{
                    valuesPtr, asserted_cast<uint32_t>(outputAccessor.count)};

                _animations._vec3.emplace_back(
                    interpolation, WHEELS_MOV(timeFrames),
                    WHEELS_MOV(valueFrames));

                concreteAnimations.push_back(
                    static_cast<void *>(&_animations._vec3.back()));
            }
            else if (outputAccessor.type == cgltf_type_vec4)
            {
                ValueAccessor<quat> valueFrames{
                    valuesPtr, asserted_cast<uint32_t>(outputAccessor.count)};

                _animations._quat.emplace_back(
                    interpolation, WHEELS_MOV(timeFrames),
                    WHEELS_MOV(valueFrames));

                concreteAnimations.push_back(
                    static_cast<void *>(&_animations._quat.back()));
            }
            else
                WHEELS_ASSERT(!"Unsupported animation output type");
        }

        for (const cgltf_animation_channel &channel :
             Span{animation.channels, animation.channels_count})
        {
            WHEELS_ASSERT(channel.target_node != nullptr);
            const uint32_t nodeIndex = asserted_cast<uint32_t>(
                cgltf_node_index(&gltfData, channel.target_node));

            WHEELS_ASSERT(channel.sampler != nullptr);
            const uint32_t samplerIndex = asserted_cast<uint32_t>(
                cgltf_animation_sampler_index(&animation, channel.sampler));

            NodeAnimations *targetAnimations = ret.find(nodeIndex);
            // These should have been initialized earlier
            WHEELS_ASSERT(targetAnimations != nullptr);
            if (channel.target_path == cgltf_animation_path_type_translation)
                targetAnimations->translation = static_cast<Animation<vec3> *>(
                    concreteAnimations[samplerIndex]);
            else if (channel.target_path == cgltf_animation_path_type_rotation)
                targetAnimations->rotation = static_cast<Animation<quat> *>(
                    concreteAnimations[samplerIndex]);
            else if (channel.target_path == cgltf_animation_path_type_scale)
                targetAnimations->scale = static_cast<Animation<vec3> *>(
                    concreteAnimations[samplerIndex]);
        }
    }

    return ret;
}

void WorldData::loadScenes(
    ScopedScratch scopeAlloc, const cgltf_data &gltfData,
    const HashMap<uint32_t, NodeAnimations> &nodeAnimations)
{
    // Parse raw nodes first so conversion to internal format happens only once
    // for potential instances
    Array<TmpNode> nodes{scopeAlloc, gltfData.nodes_count};
    for (const cgltf_node &gltfNode :
         Span{gltfData.nodes, gltfData.nodes_count})
    {
        nodes.emplace_back(gltfNode.name != nullptr ? gltfNode.name : "");
        TmpNode &node = nodes.back();

        node.children.reserve(gltfNode.children_count);
        for (const cgltf_node *child :
             Span{gltfNode.children, gltfNode.children_count})
        {
            WHEELS_ASSERT(child != nullptr);
            const cgltf_size index = cgltf_node_index(&gltfData, child);
            node.children.push_back(asserted_cast<uint32_t>(index));
        }

        if (gltfNode.mesh != nullptr)
            node.modelID = asserted_cast<uint32_t>(
                cgltf_mesh_index(&gltfData, gltfNode.mesh));
        if (gltfNode.camera != nullptr)
        {
            const uint32_t cameraIndex = asserted_cast<uint32_t>(
                cgltf_camera_index(&gltfData, gltfNode.camera));
            const cgltf_camera &cam = *gltfNode.camera;
            if (cam.type == cgltf_camera_type_perspective)
            {
                if (_cameras.size() <= cameraIndex)
                {
                    _cameras.resize(cameraIndex + 1);
                    _cameraDynamic.resize(cameraIndex + 1);
                }

                _cameras[cameraIndex] = CameraParameters{
                    .fov = static_cast<float>(cam.data.perspective.yfov),
                    .zN = static_cast<float>(cam.data.perspective.znear),
                    .zF = static_cast<float>(cam.data.perspective.zfar),
                };

                node.camera = cameraIndex;
            }
            else
                fprintf(
                    stderr, "Unsupported camera type '%s'\n",
                    sCgltfCameraTypeStr[cam.type]);
        }
        if (gltfNode.light != nullptr)
            node.light = asserted_cast<uint32_t>(
                cgltf_light_index(&gltfData, gltfNode.light));

        vec3 translation{0.f};
        vec3 scale{1.f};
        quat rotation{1.f, 0.f, 0.f, 0.f};
        if (gltfNode.has_matrix == 1)
        {
            // Spec defines the matrix to be decomposeable to T * R * S
            const mat4 matrix = make_mat4(&gltfNode.matrix[0]);
            vec3 skew;
            vec4 perspective;
            decompose(matrix, scale, rotation, translation, skew, perspective);
        }
        if (gltfNode.has_translation == 1)
            translation = make_vec3(&gltfNode.translation[0]);
        if (gltfNode.has_rotation == 1)
            rotation = make_quat(&gltfNode.rotation[0]);
        if (gltfNode.has_scale == 1)
            scale = make_vec3(&gltfNode.scale[0]);

        // Skip transform components that are close to identity
        const float srtThreshold = 0.001f;

        if (any(lessThan(translation, vec3{-srtThreshold})) ||
            any(greaterThan(translation, vec3{srtThreshold})))
            node.translation = translation;

        const vec3 eulers = eulerAngles(rotation);
        if (any(lessThan(eulers, vec3{-srtThreshold})) ||
            any(greaterThan(eulers, vec3{srtThreshold})))
            node.rotation = rotation;

        if (any(lessThan(scale, vec3{1.f - srtThreshold})) ||
            any(greaterThan(scale, vec3{1.f + srtThreshold})))
            node.scale = scale;
    }

    const cgltf_size defaultScene =
        cgltf_scene_index(&gltfData, gltfData.scene);
    _currentScene = std::max(defaultScene, (size_t)0);

    // Traverse scene trees and generate actual scene datas
    _scenes.reserve(gltfData.scenes_count);
    for (const cgltf_scene &gltfScene :
         Span{gltfData.scenes, gltfData.scenes_count})
    {
        _scenes.emplace_back();

        gatherScene(scopeAlloc.child_scope(), gltfData, gltfScene, nodes);

        Scene &scene = _scenes.back();

        // Nodes won't move in memory anymore so we can register the
        // animation targets
        for (Scene::Node &node : scene.nodes)
        {
            const NodeAnimations *animations =
                nodeAnimations.find(node.gltfSourceNode);
            if (animations != nullptr)
            {
                if (animations->translation.has_value())
                {
                    node.dynamicTransform = true;

                    Animation<vec3> *animation = *animations->translation;
                    scene.endTimeS =
                        std::max(scene.endTimeS, animation->endTimeS());

                    if (!node.translation.has_value())
                        node.translation = vec3{0.f};
                    animation->registerTarget(*node.translation);
                }
                if (animations->rotation.has_value())
                {
                    node.dynamicTransform = true;

                    Animation<quat> *animation = *animations->rotation;
                    scene.endTimeS =
                        std::max(scene.endTimeS, animation->endTimeS());

                    if (!node.rotation.has_value())
                        node.rotation = quat{1.f, 0.f, 0.f, 0.f};
                    animation->registerTarget(*node.rotation);
                }
                if (animations->scale.has_value())
                {
                    node.dynamicTransform = true;

                    Animation<vec3> *animation = *animations->scale;
                    scene.endTimeS =
                        std::max(scene.endTimeS, animation->endTimeS());

                    if (!node.scale.has_value())
                        node.scale = vec3{1.f};
                    animation->registerTarget(*node.scale);
                }
            }
            else
                // Non-animated nodes should be mapped too
                WHEELS_ASSERT(!"Node not found in animation data");
        }

        // Propagate dynamic flags
        {
            ScopedScratch scratch = scopeAlloc.child_scope();
            Array<uint32_t> nodeStack{scratch, scene.nodes.size()};
            Array<bool> parentDynamics{scratch, scene.nodes.size()};
            HashSet<uint32_t> visited{scratch, scene.nodes.size()};
            for (uint32_t rootIndex : scene.rootNodes)
            {
                nodeStack.clear();
                parentDynamics.clear();
                visited.clear();

                nodeStack.push_back(rootIndex);
                parentDynamics.push_back(false);
                while (!nodeStack.empty())
                {
                    const uint32_t nodeIndex = nodeStack.back();
                    if (visited.find(nodeIndex) != visited.end())
                    {
                        nodeStack.pop_back();
                        parentDynamics.pop_back();
                    }
                    else
                    {
                        visited.insert(nodeIndex);
                        Scene::Node &node = scene.nodes[nodeIndex];

                        const uint32_t first_child = node.firstChild;
                        const uint32_t last_child = node.lastChild;
                        for (uint32_t child = first_child; child <= last_child;
                             ++child)
                            nodeStack.push_back(child);

                        const bool parentDynamic = parentDynamics.back();
                        node.dynamicTransform |= parentDynamic;

                        if (node.dynamicTransform && node.camera.has_value())
                            _cameraDynamic[*node.camera] = true;

                        parentDynamics.emplace_back(node.dynamicTransform);
                    }
                }
            }
        }

        // Scatter random lights in the scene
        // {
        //     const vec3 minBounds{-10.f, 0.5f, -5.f};
        //     const vec3 maxBounds{10.f, 7.f, 5.f};
        //     auto rando = [](float min, float max) -> float {
        //         return std::rand() / static_cast<float>(RAND_MAX) *
        //                    (max - min) +
        //                min;
        //     };
        //     for (auto i = 0; i < 128; ++i)
        //     {
        //         // rando W -> radiance
        //         auto radiance =
        //             vec3{rando(1.f, 5.f), rando(1.f, 5.f), rando(1.f, 5.f)} /
        //             (4.f * glm::pi<float>());
        //         const auto luminance =
        //             dot(radiance, vec3{0.2126, 0.7152, 0.0722});
        //         const auto minLuminance = 0.01f;
        //         const auto radius = sqrt(luminance / minLuminance);

        //         scene.lights.pointLights.data.push_back(PointLight{
        //             .radianceAndRadius = vec4{radiance, radius},
        //             .position =
        //                 vec4{
        //                     rando(minBounds.x, maxBounds.x),
        //                     rando(minBounds.y, maxBounds.y),
        //                     rando(minBounds.z, maxBounds.z), 1.f},
        //         });
        //     }
        // }
    }

    // Make sure we always have a camera
    if (_cameras.empty())
    {
        _cameras.emplace_back();
        _cameraDynamic.push_back(false);
    }
}

void WorldData::gatherScene(
    ScopedScratch scopeAlloc, const cgltf_data &gltfData,
    const cgltf_scene &gltfScene, const Array<TmpNode> &nodes)
{
    struct NodePair
    {
        uint32_t tmpNode{0xFFFFFFFF};
        uint32_t sceneNode{0xFFFFFFFF};
    };
    Array<NodePair> nodeStack{scopeAlloc, nodes.size()};

    Scene &scene = _scenes.back();

    bool directionalLightFound = false;

    for (const cgltf_node *node : Span{gltfScene.nodes, gltfScene.nodes_count})
    {
        WHEELS_ASSERT(node != nullptr);
        const cgltf_size nodeIndex = cgltf_node_index(&gltfData, node);

        // Our node indices don't match gltf's anymore, push index of the
        // new node into roots
        scene.rootNodes.push_back(asserted_cast<uint32_t>(scene.nodes.size()));
        scene.nodes.emplace_back();
        scene.fullNodeNames.emplace_back(gAllocators.general);

        // Start adding nodes from the new root
        nodeStack.clear();
        nodeStack.emplace_back(
            asserted_cast<uint32_t>(nodeIndex),
            asserted_cast<uint32_t>(scene.nodes.size() - 1));
        while (!nodeStack.empty())
        {
            const NodePair indices = nodeStack.pop_back();
            const TmpNode &tmpNode = nodes[indices.tmpNode];

            // Push children to the back of nodes before getting the current
            // node's reference to avoid it invalidating
            const uint32_t childCount =
                asserted_cast<uint32_t>(tmpNode.children.size());
            const uint32_t firstChild =
                asserted_cast<uint32_t>(scene.nodes.size());
            // If no children, firstChild <= lastChild false as intended.
            const uint32_t lastChild = firstChild + childCount - 1;
            scene.nodes.resize(
                scene.nodes.size() + asserted_cast<size_t>(childCount));

            Scene::Node &sceneNode = scene.nodes[indices.sceneNode];
            sceneNode.gltfSourceNode = indices.tmpNode;
            sceneNode.firstChild = firstChild;
            sceneNode.lastChild = lastChild;

            // Parent initialized this with the parent 'path'
            scene.fullNodeNames[indices.sceneNode].extend(
                StrSpan{tmpNode.gltfName.data(), tmpNode.gltfName.size()});
            // This span is stable now even though the array of names is not
            sceneNode.fullName = scene.fullNodeNames[indices.sceneNode];

            for (uint32_t i = 0; i < childCount; ++i)
            {
                const uint32_t childIndex = sceneNode.firstChild + i;
                scene.nodes[childIndex].parent = indices.sceneNode;
                nodeStack.emplace_back(
                    asserted_cast<uint32_t>(tmpNode.children[i]),
                    asserted_cast<uint32_t>(childIndex));

                scene.fullNodeNames.emplace_back(
                    gAllocators.general, sceneNode.fullName);
                scene.fullNodeNames.back().push_back('/');
            }

            sceneNode.translation = tmpNode.translation;
            sceneNode.rotation = tmpNode.rotation;
            sceneNode.scale = tmpNode.scale;
            sceneNode.modelID = tmpNode.modelID;
            sceneNode.camera = tmpNode.camera;

            if (sceneNode.modelID.has_value())
            {
                sceneNode.modelInstance =
                    asserted_cast<uint32_t>(scene.modelInstances.size());
                // TODO:
                // Why is id needed here? It's just the index in the array
                scene.modelInstances.push_back(ModelInstance{
                    .id = *sceneNode.modelInstance,
                    .modelID = *sceneNode.modelID,
                    .fullName = sceneNode.fullName,
                });
                scene.drawInstanceCount += asserted_cast<uint32_t>(
                    _models[*sceneNode.modelID].subModels.size());
            }

            if (tmpNode.light.has_value())
            {
                const cgltf_light &light = gltfData.lights[*tmpNode.light];
                if (light.type == cgltf_light_type_directional)
                {
                    if (directionalLightFound)
                    {
                        fprintf(
                            stderr, "Found second directional light for a "
                                    "scene."
                                    " Ignoring since only one is supported\n");
                    }
                    auto &parameters = scene.lights.directionalLight.parameters;
                    // gltf blender exporter puts W/m^2 into intensity
                    parameters.irradiance =
                        vec4{
                            static_cast<float>(light.color[0]),
                            static_cast<float>(light.color[1]),
                            static_cast<float>(light.color[2]), 0.f} *
                        static_cast<float>(light.intensity);

                    sceneNode.directionalLight = true;
                    directionalLightFound = true;
                }
                else if (light.type == cgltf_light_type_point)
                {
                    auto radiance =
                        vec3{
                            static_cast<float>(light.color[0]),
                            static_cast<float>(light.color[1]),
                            static_cast<float>(light.color[2])} *
                        static_cast<float>(light.intensity)
                        // gltf blender exporter puts W into intensity
                        / (4.f * glm::pi<float>());
                    const auto luminance =
                        dot(radiance, vec3{0.2126, 0.7152, 0.0722});
                    const auto minLuminance = 0.01f;
                    const auto radius = light.range > 0.f
                                            ? light.range
                                            : sqrt(luminance / minLuminance);

                    sceneNode.pointLight = asserted_cast<uint32_t>(
                        scene.lights.pointLights.data.size());
                    scene.lights.pointLights.data.emplace_back();
                    auto &sceneLight = scene.lights.pointLights.data.back();

                    sceneLight.radianceAndRadius = vec4{radiance, radius};
                }
                else if (light.type == cgltf_light_type_spot)
                {
                    sceneNode.spotLight = asserted_cast<uint32_t>(
                        scene.lights.spotLights.data.size());
                    scene.lights.spotLights.data.emplace_back();
                    auto &sceneLight = scene.lights.spotLights.data.back();

                    // Angular attenuation rom gltf spec
                    const auto angleScale =
                        1.f /
                        max(0.001f, static_cast<float>(
                                        cos(light.spot_inner_cone_angle) -
                                        cos(light.spot_outer_cone_angle)));
                    const auto angleOffset =
                        static_cast<float>(-cos(light.spot_outer_cone_angle)) *
                        angleScale;

                    sceneLight.radianceAndAngleScale =
                        vec4{
                            static_cast<float>(light.color[0]),
                            static_cast<float>(light.color[1]),
                            static_cast<float>(light.color[2]), 0.f} *
                        static_cast<float>(light.intensity);
                    // gltf blender exporter puts W into intensity
                    sceneLight.radianceAndAngleScale /= 4.f * glm::pi<float>();
                    sceneLight.radianceAndAngleScale.w = angleScale;

                    sceneLight.positionAndAngleOffset.w = angleOffset;
                }
                else
                    fprintf(
                        stderr, "Unsupported light type '%s'\n",
                        sCgltfLightTypeStr[light.type]);
            }
        }
    }

    // Honor scene lighting
    if (!directionalLightFound && (!scene.lights.pointLights.data.empty() ||
                                   !scene.lights.spotLights.data.empty()))
    {
        scene.lights.directionalLight.parameters.irradiance = vec4{0.f};
    }
}

void WorldData::createBuffers()
{
    for (size_t i = 0; i < _materialsBuffers.capacity(); ++i)
        _materialsBuffers[i] = gDevice.createBuffer(BufferCreateInfo{
            .desc =
                BufferDescription{
                    .byteSize = _materials.size() * sizeof(_materials[0]),
                    .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                    .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent,
                },
            .initialData = _materials.data(),
            .debugName = "MaterialsBuffer",
        });

    {
        size_t maxModelInstanceTransforms = 0;
        for (auto &scene : _scenes)
        {
            maxModelInstanceTransforms = std::max(
                maxModelInstanceTransforms, scene.modelInstances.size());

            scene.drawInstancesBuffer = gDevice.createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize = sizeof(Scene::DrawInstance) *
                                    scene.drawInstanceCount,
                        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent,
                    },
                .debugName = "DrawInstances",
            });
        }

        // Make room for one extra frame because the previous frame's transforms
        // are read for motion
        const uint32_t bufferSize = asserted_cast<uint32_t>(
            ((maxModelInstanceTransforms * sizeof(ModelInstance::Transforms) +
              static_cast<size_t>(RingBuffer::sAlignment)) +
             (maxModelInstanceTransforms * sizeof(float) +
              static_cast<size_t>(RingBuffer::sAlignment))) *
            (MAX_FRAMES_IN_FLIGHT + 1));
        _modelInstanceTransformsRing.init(
            vk::BufferUsageFlagBits::eStorageBuffer, bufferSize,
            "ModelInstanceTransformRing");
    }
}

void WorldData::reflectBindings(ScopedScratch scopeAlloc)
{
    const auto reflect =
        [&](const String &defines, const std::filesystem::path &relPath)
    {
        Optional<ShaderReflection> compResult = gDevice.reflectShader(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = relPath,
                .defines = defines,
            },
            true);
        if (!compResult.has_value())
            throw std::runtime_error(
                std::string("Failed to create reflection for '") +
                relPath.string() + '\'');

        return WHEELS_MOV(*compResult);
    };

    {
        WHEELS_ASSERT(!_samplers.empty());
        _dsLayouts.materialSamplerCount =
            asserted_cast<uint32_t>(_samplers.size());

        const size_t len = 192;
        String defines{scopeAlloc, len};
        appendDefineStr(
            defines, "MATERIAL_DATAS_SET", sMaterialDatasReflectionSet);
        appendDefineStr(
            defines, "MATERIAL_TEXTURES_SET", sMaterialTexturesReflectionSet);
        appendDefineStr(
            defines, "NUM_MATERIAL_SAMPLERS", _dsLayouts.materialSamplerCount);
        defines.extend("#extension GL_EXT_nonuniform_qualifier : require\n");
        WHEELS_ASSERT(defines.size() <= len);

        _materialsReflection = reflect(defines, "shader/scene/materials.glsl");
    }

    {
        const size_t len = 169;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "GEOMETRY_SET", sGeometryReflectionSet);
        defines.extend("#extension GL_EXT_nonuniform_qualifier : require\n");
        defines.extend("#extension GL_EXT_shader_16bit_storage : require\n");
        defines.extend("#extension GL_EXT_shader_8bit_storage : require\n");
        WHEELS_ASSERT(defines.size() <= len);

        _geometryReflection = reflect(defines, "shader/scene/geometry.glsl");
    }

    {
        const size_t len = 64;
        String defines{scopeAlloc, len};
        appendDefineStr(
            defines, "SCENE_INSTANCES_SET", sSceneInstancesReflectionSet);
        WHEELS_ASSERT(defines.size() <= len);

        _sceneInstancesReflection =
            reflect(defines, "shader/scene/instances.glsl");
    }

    {
        const size_t len = 92;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "LIGHTS_SET", sLightsReflectionSet);
        PointLights::appendShaderDefines(defines);
        SpotLights::appendShaderDefines(defines);
        WHEELS_ASSERT(defines.size() <= len);

        _lightsReflection = reflect(defines, "shader/scene/lights.glsl");
    }

    {
        const size_t len = 32;
        String defines{scopeAlloc, len};
        appendDefineStr(defines, "SKYBOX_SET", sSkyboxReflectionSet);
        WHEELS_ASSERT(defines.size() <= len);

        _skyboxReflection = reflect(defines, "shader/scene/skybox.glsl");
    }
}

void WorldData::createDescriptorSets(
    ScopedScratch scopeAlloc, const RingBuffers &ringBuffers)
{
    WHEELS_ASSERT(ringBuffers.constantsRing != nullptr);
    WHEELS_ASSERT(ringBuffers.lightDataRing != nullptr);

    WHEELS_ASSERT(_materialsReflection.has_value());
    _dsLayouts.materialDatas = _materialsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), sMaterialDatasReflectionSet,
        vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eAnyHitKHR);

    {
        const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT>
            materialDatasLayouts{_dsLayouts.materialDatas};
        const StaticArray<const char *, MAX_FRAMES_IN_FLIGHT> debugNames{
            "MaterialDatas"};
        _descriptorAllocator.allocate(
            materialDatasLayouts, debugNames,
            _descriptorSets.materialDatas.mut_span());
    }

    WHEELS_ASSERT(_materialsBuffers.size() == MAX_FRAMES_IN_FLIGHT);
    WHEELS_ASSERT(_descriptorSets.materialDatas.size() == MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        const StaticArray descriptorInfos{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = _materialsBuffers[i].handle,
                .range = VK_WHOLE_SIZE,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = ringBuffers.constantsRing->buffer(),
                .range = sizeof(float),
            }},
        }};
        const Array descriptorWrites =
            _materialsReflection->generateDescriptorWrites(
                scopeAlloc, sMaterialDatasReflectionSet,
                _descriptorSets.materialDatas[i], descriptorInfos);
        gDevice.logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    {
        Array<vk::DescriptorImageInfo> materialSamplerInfos{
            scopeAlloc, _samplers.size()};
        for (const auto &s : _samplers)
            materialSamplerInfos.push_back(
                vk::DescriptorImageInfo{.sampler = s});
        const auto samplerInfoCount =
            asserted_cast<uint32_t>(materialSamplerInfos.size());
        _dsLayouts.materialSamplerCount = samplerInfoCount;

        // Use capacity instead of size so that this allocates descriptors for
        // textures that are loaded later
        Array<vk::DescriptorImageInfo> materialImageInfos{
            scopeAlloc, _texture2Ds.capacity()};
        // Fill missing textures with the default info so potential reads
        // are still to valid descriptors
        WHEELS_ASSERT(_texture2Ds.size() == 1);
        const vk::DescriptorImageInfo defaultInfo = _texture2Ds[0].imageInfo();
        for (size_t i = 0; i < materialImageInfos.capacity(); ++i)
            materialImageInfos.push_back(defaultInfo);

        const auto imageInfoCount =
            asserted_cast<uint32_t>(materialImageInfos.size());

        const StaticArray bindingFlags{{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
                // Texture bindings for deferred loads are updated before frame
                // cb submission, for textures that aren't accessed by any frame
                // in flight
                vk::DescriptorBindingFlagBits::ePartiallyBound |
                vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending},
        }};

        WHEELS_ASSERT(_materialsReflection.has_value());
        _dsLayouts.materialTextures =
            _materialsReflection->createDescriptorSetLayout(
                scopeAlloc.child_scope(), sMaterialTexturesReflectionSet,
                vk::ShaderStageFlagBits::eFragment |
                    vk::ShaderStageFlagBits::eRaygenKHR |
                    vk::ShaderStageFlagBits::eAnyHitKHR,
                Span{&imageInfoCount, 1}, bindingFlags);

        _descriptorSets.materialTextures = _descriptorAllocator.allocate(
            _dsLayouts.materialTextures, "MaterialTextures", imageInfoCount);

        const StaticArray descriptorInfos{{
            DescriptorInfo{materialSamplerInfos},
            DescriptorInfo{materialImageInfos},
        }};

        const Array descriptorWrites =
            _materialsReflection->generateDescriptorWrites(
                scopeAlloc, sMaterialTexturesReflectionSet,
                _descriptorSets.materialTextures, descriptorInfos);
        gDevice.logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);

        _deferredLoadingContext->textureArrayBinding =
            asserted_cast<uint32_t>(materialSamplerInfos.size());
    }

    {
        // Geometry layouts and descriptor set
        const uint32_t bufferCount = 2 + sMaxGeometryBuffersCount;

        const StaticArray bindingFlags{{
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{},
            vk::DescriptorBindingFlags{
                vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
                vk::DescriptorBindingFlagBits::ePartiallyBound},
        }};

        WHEELS_ASSERT(_geometryReflection.has_value());
        _dsLayouts.geometry = _geometryReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), sGeometryReflectionSet,
            vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eCompute |
                vk::ShaderStageFlagBits::eRaygenKHR |
                vk::ShaderStageFlagBits::eAnyHitKHR |
                vk::ShaderStageFlagBits::eMeshEXT,
            Span{&bufferCount, 1}, bindingFlags);

        WHEELS_ASSERT(_descriptorSets.geometry.size() == MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            _descriptorSets.geometry[i] = _descriptorAllocator.allocate(
                _dsLayouts.geometry, "Geometry", bufferCount);

            const StaticArray descriptorInfos{{
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _geometryMetadatasBuffers[i].handle,
                    .range = VK_WHOLE_SIZE,
                }},
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _meshletCountsBuffers[i].handle,
                    .range = VK_WHOLE_SIZE,
                }},
                DescriptorInfo{Span<const vk::DescriptorBufferInfo>{}},
            }};

            const Array descriptorWrites =
                _geometryReflection->generateDescriptorWrites(
                    scopeAlloc, sGeometryReflectionSet,
                    _descriptorSets.geometry[i], descriptorInfos);

            gDevice.logical().updateDescriptorSets(
                asserted_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(), 0, nullptr);
        }
    }

    // RT layout
    {
        // TODO:
        // Need to support differing flags for binds within set here? Does AMD
        // support binding AS in stages other than raygen (recursion = 1)? Is
        // perf affected if AS is bound but unused in anyhit?
        const StaticArray layoutBindings{{
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR |
                              vk::ShaderStageFlagBits::eAnyHitKHR,
            },
        }};
        const vk::DescriptorSetLayoutCreateInfo createInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        };
        _dsLayouts.rayTracing =
            gDevice.logical().createDescriptorSetLayout(createInfo);
    }

    WHEELS_ASSERT(_sceneInstancesReflection.has_value());
    _dsLayouts.sceneInstances =
        _sceneInstancesReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), sSceneInstancesReflectionSet,
            vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eCompute |
                vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eRaygenKHR |
                vk::ShaderStageFlagBits::eAnyHitKHR |
                vk::ShaderStageFlagBits::eMeshEXT);

    WHEELS_ASSERT(_lightsReflection.has_value());
    _dsLayouts.lights = _lightsReflection->createDescriptorSetLayout(
        scopeAlloc.child_scope(), sLightsReflectionSet,
        vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eRaygenKHR);

    // Per light type
    {
        _descriptorSets.lights =
            _descriptorAllocator.allocate(_dsLayouts.lights, "Lights");

        const StaticArray lightInfos{{
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = ringBuffers.lightDataRing->buffer(),
                .offset = 0,
                .range = sizeof(DirectionalLight::Parameters),
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = ringBuffers.lightDataRing->buffer(),
                .offset = 0,
                .range = PointLights::sBufferByteSize,
            }},
            DescriptorInfo{vk::DescriptorBufferInfo{
                .buffer = ringBuffers.lightDataRing->buffer(),
                .offset = 0,
                .range = SpotLights::sBufferByteSize,
            }},
        }};

        const Array descriptorWrites =
            _lightsReflection->generateDescriptorWrites(
                scopeAlloc, sLightsReflectionSet, _descriptorSets.lights,
                lightInfos);

        gDevice.logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    // Scene descriptor sets
    const size_t sceneCount = _scenes.size();
    for (size_t i = 0; i < sceneCount; ++i)
    {
        Scene &scene = _scenes[i];
        {
            scene.sceneInstancesDescriptorSet = _descriptorAllocator.allocate(
                _dsLayouts.sceneInstances, "SceneInstances");

            const StaticArray descriptorInfos{{
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _modelInstanceTransformsRing.buffer(),
                    .range = scene.modelInstances.size() *
                             sizeof(ModelInstance::Transforms),
                }},
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _modelInstanceTransformsRing.buffer(),
                    .range = scene.modelInstances.size() *
                             sizeof(ModelInstance::Transforms),
                }},
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = _modelInstanceTransformsRing.buffer(),
                    .range = scene.modelInstances.size() * sizeof(float),
                }},
                DescriptorInfo{vk::DescriptorBufferInfo{
                    .buffer = scene.drawInstancesBuffer.handle,
                    .range = VK_WHOLE_SIZE,
                }},
            }};
            const Array descriptorWrites =
                _sceneInstancesReflection->generateDescriptorWrites(
                    scopeAlloc, sSceneInstancesReflectionSet,
                    scene.sceneInstancesDescriptorSet, descriptorInfos);

            gDevice.logical().updateDescriptorSets(
                asserted_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(), 0, nullptr);
        }
        {
            scene.rtDescriptorSet =
                _descriptorAllocator.allocate(_dsLayouts.rayTracing, "Rt");
            // DS is written by World::Impl when the TLAS is created
        }
    }

    // Skybox layout and descriptor set
    {
        WHEELS_ASSERT(_skyboxReflection.has_value());
        _dsLayouts.skybox = _skyboxReflection->createDescriptorSetLayout(
            scopeAlloc.child_scope(), sSkyboxReflectionSet,
            vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eCompute |
                vk::ShaderStageFlagBits::eRaygenKHR);

        _descriptorSets.skybox =
            _descriptorAllocator.allocate(_dsLayouts.skybox, "Skybox");

        const StaticArray descriptorInfos{{
            DescriptorInfo{_skyboxResources.texture.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _skyboxResources.sampler,
                .imageView = _skyboxResources.irradiance.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _skyboxResources.sampler,
                .imageView = _skyboxResources.specularBrdfLut.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
            DescriptorInfo{vk::DescriptorImageInfo{
                .sampler = _skyboxResources.sampler,
                .imageView = _skyboxResources.radiance.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            }},
        }};
        const Array descriptorWrites =
            _skyboxReflection->generateDescriptorWrites(
                scopeAlloc, sSkyboxReflectionSet, _descriptorSets.skybox,
                descriptorInfos);

        gDevice.logical().updateDescriptorSets(
            asserted_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }
}

bool WorldData::pollMeshWorker(vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    WHEELS_ASSERT(ctx.loadedMeshCount < ctx.meshes.size());

    bool newMeshLoaded = false;
    const size_t maxMeshesPerFrame = 10;
    for (size_t i = 0; i < maxMeshesPerFrame; ++i)
    {
        // Let's pop meshes one by one to potentially let the async worker push
        // new ones to fill the quota while we're in this loop
        Optional<Pair<UploadedGeometryData, MeshInfo>> loaded;
        {
            const std::lock_guard _lock{ctx.loadedMeshesMutex};
            if (ctx.loadedMeshes.empty())
                break;

            loaded = ctx.loadedMeshes.front();
            ctx.loadedMeshes.erase(0);
        }

        if (loaded.has_value())
        {
            newMeshLoaded = true;
            {
                const std::lock_guard _lock{ctx.geometryBuffersMutex};
                // Copy over any newly created geometry buffers
                while (_geometryBuffers.size() < ctx.geometryBuffers.size())
                {
                    _geometryBuffers.push_back(
                        ctx.geometryBuffers[_geometryBuffers.size()].clone());
                    WHEELS_ASSERT(
                        _geometryBuffers.size() <= sMaxGeometryBuffersCount &&
                        "The layout requires a hard limit on the max number of "
                        "geometry buffers");
                    _geometryBufferAllocatedByteCounts.push_back(0u);
                }
            }

            const UploadedGeometryData &uploadedData = loaded->first;
            const MeshInfo &info = loaded->second;
            const uint32_t targetBufferI = uploadedData.metadata.bufferIndex;
            WHEELS_ASSERT(uploadedData.byteCount > 0);

            const uint32_t previousAllocatedByteCount =
                _geometryBufferAllocatedByteCounts[targetBufferI];
            WHEELS_ASSERT(
                ((uploadedData.metadata.usesShortIndices &&
                  previousAllocatedByteCount ==
                      uploadedData.metadata.indicesOffset * sizeof(uint16_t)) ||
                 (previousAllocatedByteCount ==
                  uploadedData.metadata.indicesOffset * sizeof(uint32_t))) &&
                "Uploaded data ranges have to be tight for valid ownership "
                "transfer");

            const QueueFamilies &families = gDevice.queueFamilies();
            WHEELS_ASSERT(families.graphicsFamily.has_value());
            WHEELS_ASSERT(families.transferFamily.has_value());

            if (*families.graphicsFamily != *families.transferFamily)
            {
                const Buffer &geometryBuffer = _geometryBuffers[targetBufferI];

                const vk::BufferMemoryBarrier2 acquireBarrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                    .srcAccessMask = vk::AccessFlagBits2::eNone,
                    .dstStageMask =
                        vk::PipelineStageFlagBits2::eVertexShader |
                        vk::PipelineStageFlagBits2::eRayTracingShaderKHR |
                        vk::PipelineStageFlagBits2::eMeshShaderEXT,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .srcQueueFamilyIndex = *families.transferFamily,
                    .dstQueueFamilyIndex = *families.graphicsFamily,
                    .buffer = geometryBuffer.handle,
                    .offset = uploadedData.byteOffset,
                    .size = uploadedData.byteCount,
                };

                cb.pipelineBarrier2(vk::DependencyInfo{
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &acquireBarrier,
                });
            }

            _geometryMetadatas[ctx.loadedMeshCount] = uploadedData.metadata;
            _meshInfos[ctx.loadedMeshCount] = info;
            _meshNames.emplace_back(gAllocators.general, uploadedData.meshName);
            // Track the used (and ownership transferred) range
            _geometryBufferAllocatedByteCounts[targetBufferI] +=
                uploadedData.byteCount;

            ctx.loadedMeshCount++;
            ctx.geometryGeneration++;
        }
    }

    return newMeshLoaded;
}

size_t WorldData::pollTextureWorker(vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    WHEELS_ASSERT(ctx.loadedImageCount < ctx.gltfData->images_count);

    size_t newTexturesLoaded = 0;
    const size_t maxTexturesPerFrame = 10;
    for (size_t i = 0; i < maxTexturesPerFrame; ++i)
    {
        bool newTextureLoaded = false;
        {
            // Let's pop textures one by one to potentially let the async worker
            // push new ones to fill the quota while we're in this loop
            const std::lock_guard _lock{ctx.loadedTexturesMutex};
            if (ctx.loadedTextures.empty())
                break;

            _texture2Ds.emplace_back(WHEELS_MOV(ctx.loadedTextures.front()));
            ctx.loadedTextures.erase(0);
            newTextureLoaded = true;
        }

        if (newTextureLoaded)
        {
            newTexturesLoaded++;

            const QueueFamilies &families = gDevice.queueFamilies();
            WHEELS_ASSERT(families.graphicsFamily.has_value());
            WHEELS_ASSERT(families.transferFamily.has_value());

            if (*families.graphicsFamily != *families.transferFamily)
            {
                const vk::ImageMemoryBarrier2 acquireBarrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                    .srcAccessMask = vk::AccessFlagBits2::eNone,
                    .dstStageMask =
                        vk::PipelineStageFlagBits2::eFragmentShader |
                        vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = *families.transferFamily,
                    .dstQueueFamilyIndex = *families.graphicsFamily,
                    .image = _texture2Ds.back().nativeHandle(),
                    .subresourceRange =
                        vk::ImageSubresourceRange{
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount = VK_REMAINING_ARRAY_LAYERS,
                        },
                };
                cb.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &acquireBarrier,
                });
            }
        }
    }

    return newTexturesLoaded;
}

void WorldData::updateDescriptorsWithNewTextures(size_t newTextureCount)
{
    WHEELS_ASSERT(_deferredLoadingContext.has_value());

    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    const size_t textureCount = _texture2Ds.size();
    for (size_t i = 0; i < newTextureCount; ++i)
    {
        const vk::DescriptorImageInfo imageInfo =
            _texture2Ds[textureCount - newTextureCount + i].imageInfo();
        const vk::WriteDescriptorSet descriptorWrite{
            .dstSet = _descriptorSets.materialTextures,
            .dstBinding = ctx.textureArrayBinding,
            // loadedImageCount is gltf images so bump by one to take our
            // default texture into account
            .dstArrayElement = ctx.loadedImageCount + 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &imageInfo,
        };
        gDevice.logical().updateDescriptorSets(1, &descriptorWrite, 0, nullptr);

        ctx.loadedImageCount++;
    }
}

bool WorldData::updateMaterials()
{
    DeferredLoadingContext &ctx = *_deferredLoadingContext;
    // Update next material(s) in line if the required textures are
    // loaded
    bool materialsUpdated = false;
    for (size_t i = ctx.loadedMaterialCount; i < ctx.materials.size(); ++i)
    {
        const Material &material = ctx.materials[i];
        const uint32_t baseColorIndex = material.baseColor.texture();
        const uint32_t normalIndex = material.normal.texture();
        const uint32_t metallicRoughnessIndex =
            material.metallicRoughness.texture();
        // Inclusive as 0 is our default, starting gltf indices from 1
        if (baseColorIndex <= ctx.loadedImageCount &&
            normalIndex <= ctx.loadedImageCount &&
            metallicRoughnessIndex <= ctx.loadedImageCount)
        {
            // These are gltf material indices so we have to take our
            // default material into account
            _materials[i + 1] = material;
            ctx.loadedMaterialCount++;
            materialsUpdated = true;
        }
        else
            break;
    }

    if (materialsUpdated)
        ctx.materialsGeneration++;

    return materialsUpdated;
}
