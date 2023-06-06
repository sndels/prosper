#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Profiler.hpp"
#include "Scene.hpp"
#include "Texture.hpp"

#include <filesystem>
#include <mutex>
#include <tiny_gltf.h>
#include <vulkan/vulkan_hash.hpp>

#include <wheels/allocators/cstdlib_allocator.hpp>
#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/hash_map.hpp>

namespace tinygltf
{
class Model;
};

class Device;
struct DeferredLoadingContext;

class World
{
  public:
    struct DSLayouts
    {
        uint32_t materialSamplerCount{0};
        vk::DescriptorSetLayout materialTextures;
        vk::DescriptorSetLayout geometry;
        vk::DescriptorSetLayout modelInstances;
        vk::DescriptorSetLayout rayTracing;
        vk::DescriptorSetLayout lights;
        vk::DescriptorSetLayout skybox;
        vk::DescriptorSetLayout skyboxOnly;
    };
    World(
        wheels::ScopedScratch scopeAlloc, Device *device,
        const std::filesystem::path &scene, bool deferredLoading);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    void handleDeferredLoading(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, Profiler &profiler);

    [[nodiscard]] const Scene &currentScene() const;
    void updateUniformBuffers(
        const Camera &cam, uint32_t nextFrame,
        wheels::ScopedScratch scopeAlloc) const;
    void drawSkybox(const vk::CommandBuffer &buffer) const;

    wheels::CstdlibAllocator _generalAlloc;
    wheels::LinearAllocator _linearAlloc;

    std::filesystem::path _sceneDir;

    // TODO: Private?
    TextureCubemap _skyboxTexture;
    Buffer _skyboxVertexBuffer;

    wheels::HashMap<Scene::Node *, CameraParameters> _cameras{_generalAlloc};
    wheels::Array<vk::Sampler> _samplers{_generalAlloc};
    wheels::Array<Texture2D> _texture2Ds{_generalAlloc};
    wheels::Array<Buffer> _geometryBuffers{_generalAlloc};
    Buffer _meshBuffersBuffer;
    wheels::Array<Material> _materials{_generalAlloc};
    wheels::Array<MeshBuffers> _meshBuffers{_generalAlloc};
    wheels::Array<MeshInfo> _meshInfos{_generalAlloc};
    wheels::Array<Buffer> _modelInstances{_generalAlloc};
    wheels::Array<AccelerationStructure> _blases{_generalAlloc};
    wheels::Array<AccelerationStructure> _tlases{_generalAlloc};
    wheels::Array<Model> _models{_generalAlloc};
    wheels::Array<Scene::Node> _nodes{_generalAlloc};
    wheels::Array<Scene> _scenes{_generalAlloc};
    size_t _currentScene{0};

    Buffer _materialsBuffer;
    vk::DescriptorSet _materialTexturesDS;
    vk::DescriptorSet _geometryDS;
    DSLayouts _dsLayouts;

    wheels::Array<Buffer> _skyboxUniformBuffers{_generalAlloc};
    wheels::Array<vk::DescriptorSet> _skyboxDSs{_generalAlloc};
    vk::DescriptorSet _skyboxOnlyDS;

    struct DeferredLoadingContext
    {
        // TODO:
        // Rough outline:
        // - At load time
        //   - Texture descriptors are initialized to the default texture
        //     - Removes need to allocate descriptors at runtime, we already
        //       know the size we want
        //   - Materials point to default tex, default sampler
        //     - Separate gpu-side material buffer and DS per in-flight frame
        //       - Generation id for material array so we can track which
        //         gpu buffers are up to date and which aren't
        //     - No need to have separate texture descriptors for in-flight
        //       frames if frames don't access descriptors until they are loaded
        //       in
        //       - Need to check if this requires some flags for the shared
        //         texture descriptors since the content will be written to
        //         while they are bound
        //   - AssyncLoadContext is initialized
        // - Beginning of frame
        //     - Transfer next texture, if loadedImageCount!= totalImages
        //       - Image loading could be async on transfer if mips were
        //         pre-generated or on compute if mips were generated on
        //         compute. The current blitting requires a graphics queue and
        //         we only might have one of them so can't do proper async.
        //     - Write new texture descriptor
        //     - For remaining unloaded materials
        //       - If all textures transferred, overwrite main materials
        //         info
        //       - else break since we only store the number of loaded materials
        //     - If new materials written, bump materials gen
        //     - Check if current frame's gpu-side materials buffer gen matches
        //       current main materials gen
        //       - memcpy new stuff if not

        DeferredLoadingContext(
            wheels::Allocator &alloc, Device *device,
            const tinygltf::Model &gltfModel);
        ~DeferredLoadingContext();

        Device *device{nullptr};
        tinygltf::Model gltfModel;
        uint32_t framesSinceFinish{0};
        uint32_t textureArrayBinding{0};
        uint32_t loadedImageCount{0};
        uint32_t loadedMaterialCount{0};
        wheels::Array<Material> materials;
        wheels::StaticArray<Buffer, MAX_FRAMES_IN_FLIGHT> stagingBuffers;
    };
    wheels::Optional<DeferredLoadingContext> _deferredLoadingContext;

  private:
    void loadTextures(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel,
        wheels::Array<Texture2DSampler> &texture2DSamplers,
        bool deferredLoading);
    void loadMaterials(
        const tinygltf::Model &gltfModel,
        const wheels::Array<Texture2DSampler> &texture2DSamplers,
        bool deferredLoading);
    void loadModels(const tinygltf::Model &gltfModel);
    void loadScenes(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel);
    void createBlases();
    void createTlases(wheels::ScopedScratch scopeAlloc);
    void createBuffers();
    void createDescriptorSets(wheels::ScopedScratch scopeAlloc);

    Device *_device{nullptr};
    DescriptorAllocator _descriptorAllocator;
};

#endif // PROSPER_WORLD_HPP
