#ifndef PROSPER_WORLD_HPP
#define PROSPER_WORLD_HPP

#include "Material.hpp"
#include "Mesh.hpp"
#include "Model.hpp"
#include "Scene.hpp"
#include "Texture.hpp"

#include <filesystem>
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
        uint32_t swapImageCount, const std::filesystem::path &scene);
    ~World();

    World(const World &other) = delete;
    World(World &&other) = delete;
    World &operator=(const World &other) = delete;
    World &operator=(World &&other) = delete;

    [[nodiscard]] const Scene &currentScene() const;
    void updateUniformBuffers(
        const Camera &cam, uint32_t nextImage,
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
    struct Texture2DSampler
    {
        uint32_t texture{0};
        uint32_t sampler{0};
    };
    wheels::Array<Texture2DSampler> _texture2DSamplers{_generalAlloc};
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

  private:
    void loadTextures(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel);
    void loadMaterials(const tinygltf::Model &gltfModel);
    void loadModels(const tinygltf::Model &gltfModel);
    void loadScenes(
        wheels::ScopedScratch scopeAlloc, const tinygltf::Model &gltfModel);
    void createBlases();
    void createTlases(wheels::ScopedScratch scopeAlloc);
    void createBuffers(uint32_t swapImageCount);
    void createDescriptorSets(
        wheels::ScopedScratch scopeAlloc, uint32_t swapImageCount);

    Device *_device{nullptr};
    DescriptorAllocator _descriptorAllocator;
};

#endif // PROSPER_WORLD_HPP
