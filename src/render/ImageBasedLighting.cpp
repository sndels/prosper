#include "ImageBasedLighting.hpp"

using namespace wheels;
using namespace glm;

namespace
{

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(
        defines, "OUT_RESOLUTION", World::sSkyboxIrradianceResolution);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/ibl/sample_irradiance.comp",
        .debugName = String{alloc, "SampleIrradianceCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

ImageBasedLighting::ImageBasedLighting(
    ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc)

: _device{device}
, _sampleIrradiance{
      WHEELS_MOV(scopeAlloc), device, staticDescriptorsAlloc,
      shaderDefinitionCallback}
{
    WHEELS_ASSERT(_device != nullptr);
}

bool ImageBasedLighting::isGenerated() const { return _generated; }

void ImageBasedLighting::recompileShaders(wheels::ScopedScratch scopeAlloc)
{
    _sampleIrradiance.recompileShader(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback);
}

void ImageBasedLighting::recordGeneration(
    vk::CommandBuffer cb, World &world, uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    {
        const StaticArray descriptorInfos{
            DescriptorInfo{world._skyboxTexture.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = world._skyboxIrradiance.view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        };
        _sampleIrradiance.updateDescriptorSet(nextFrame, descriptorInfos);

        world._skyboxIrradiance.transition(cb, ImageState::ComputeShaderWrite);

        const auto _s = profiler->createCpuGpuScope(cb, "SampleIrradiance");

        const uvec3 groups = uvec3{
            (uvec2{World::sSkyboxIrradianceResolution} - 1u) / 16u + 1u, 6u};

        const vk::DescriptorSet storageSet =
            _sampleIrradiance.storageSet(nextFrame);
        _sampleIrradiance.record(cb, groups, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition for
        // all users
        world._skyboxIrradiance.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }
    _generated = true;
}
