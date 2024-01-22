#ifndef PROSPER_SCENE_WORLD_RENDER_STRUCTS_HPP
#define PROSPER_SCENE_WORLD_RENDER_STRUCTS_HPP

#include "../gfx/Resources.hpp"
#include "Texture.hpp"
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/static_array.hpp>

struct WorldDSLayouts
{
    uint32_t materialSamplerCount{0};
    vk::DescriptorSetLayout materialDatas;
    vk::DescriptorSetLayout materialTextures;
    vk::DescriptorSetLayout geometry;
    vk::DescriptorSetLayout modelInstances;
    vk::DescriptorSetLayout rayTracing;
    vk::DescriptorSetLayout lights;
    vk::DescriptorSetLayout skybox;
};

struct WorldByteOffsets
{
    uint32_t modelInstanceTransforms{0};
    uint32_t previousModelInstanceTransforms{0};
    uint32_t modelInstanceScales{0};
    uint32_t directionalLight{0};
    uint32_t pointLights{0};
    uint32_t spotLights{0};
    uint32_t globalMaterialConstants{0};
};

struct WorldDescriptorSets
{
    vk::DescriptorSet lights;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> materialDatas;
    vk::DescriptorSet materialTextures;
    wheels::StaticArray<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> geometry;
    vk::DescriptorSet skybox;
};

struct SkyboxResources
{
    static const uint32_t sSkyboxIrradianceResolution = 64;
    static const uint32_t sSpecularBrdfLutResolution = 512;
    static const uint32_t sSkyboxRadianceResolution = 512;

    TextureCubemap texture;
    Image irradiance;
    Image specularBrdfLut;
    Image radiance;
    wheels::Array<vk::ImageView> radianceViews;
    Buffer vertexBuffer;
    vk::Sampler sampler;
};

#endif // PROSPER_SCENE_WORLD_RENDER_STRUCTS_HPP
