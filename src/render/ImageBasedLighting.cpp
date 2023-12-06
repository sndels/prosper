#include "ImageBasedLighting.hpp"

#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include <glm/detail/type_vec2.hpp>
#include <glm/fwd.hpp>

using namespace wheels;
using namespace glm;

namespace
{

ComputePass::Shader sampleIrradianceShaderDefinitionCallback(Allocator &alloc)
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

ComputePass::Shader integrateSpecularBrdfShaderDefinitionCallback(
    Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(
        defines, "OUT_RESOLUTION", World::sSpecularBrdfLutResolution);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/ibl/integrate_specular_brdf.comp",
        .debugName = String{alloc, "IntegrateSpecularBrdfCS"},
        .defines = WHEELS_MOV(defines),
    };
}

struct PrefilterRadiancePC
{
    uint mipCount{0};
};

ComputePass::Shader prefilterRadianceShaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(
        defines, "OUT_RESOLUTION", World::sSkyboxRadianceResolution);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/ibl/prefilter_radiance.comp",
        .debugName = String{alloc, "PrefilterRadianceCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

ImageBasedLighting::ImageBasedLighting(
    ScopedScratch scopeAlloc, Device *device,
    DescriptorAllocator *staticDescriptorsAlloc)

: _device{device}
, _sampleIrradiance{scopeAlloc.child_scope(), device, staticDescriptorsAlloc, sampleIrradianceShaderDefinitionCallback}
, _integrateSpecularBrdf{scopeAlloc.child_scope(), device, staticDescriptorsAlloc, integrateSpecularBrdfShaderDefinitionCallback}
, _prefilterRadiance{
      scopeAlloc.child_scope(),
      device,
      staticDescriptorsAlloc,
      prefilterRadianceShaderDefinitionCallback,
  }
{
    WHEELS_ASSERT(_device != nullptr);
}

bool ImageBasedLighting::isGenerated() const { return _generated; }

void ImageBasedLighting::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    _sampleIrradiance.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        sampleIrradianceShaderDefinitionCallback);
    _integrateSpecularBrdf.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        integrateSpecularBrdfShaderDefinitionCallback);
    _prefilterRadiance.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        prefilterRadianceShaderDefinitionCallback);
}

void ImageBasedLighting::recordGeneration(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
    uint32_t nextFrame, Profiler *profiler)
{
    WHEELS_ASSERT(profiler != nullptr);

    SkyboxResources &skyboxResources = world.skyboxResources();

    {
        const StaticArray descriptorInfos{
            DescriptorInfo{skyboxResources.texture.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = skyboxResources.irradiance.view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        };
        _sampleIrradiance.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        skyboxResources.irradiance.transition(
            cb, ImageState::ComputeShaderWrite);

        const auto _s = profiler->createCpuGpuScope(cb, "SampleIrradiance");

        const uvec3 groups = uvec3{
            (uvec2{World::sSkyboxIrradianceResolution} - 1u) / 16u + 1u, 6u};

        const vk::DescriptorSet storageSet =
            _sampleIrradiance.storageSet(nextFrame);
        _sampleIrradiance.record(cb, groups, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition for
        // all users
        skyboxResources.irradiance.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }

    {
        const StaticArray descriptorInfos{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = skyboxResources.specularBrdfLut.view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        };
        _integrateSpecularBrdf.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        skyboxResources.specularBrdfLut.transition(
            cb, ImageState::ComputeShaderWrite);

        const auto _s =
            profiler->createCpuGpuScope(cb, "IntegrateSpecularBrdf");

        const uvec3 groups = uvec3{
            (uvec2{World::sSpecularBrdfLutResolution} - 1u) / 16u + 1u, 1u};

        const vk::DescriptorSet storageSet =
            _integrateSpecularBrdf.storageSet(nextFrame);
        _integrateSpecularBrdf.record(cb, groups, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition for
        // all users
        skyboxResources.specularBrdfLut.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }

    {
        const uint32_t mipCount = skyboxResources.radiance.mipCount;
        WHEELS_ASSERT(mipCount == skyboxResources.radianceViews.size());

        StaticArray<vk::DescriptorImageInfo, 15> imageInfos{
            vk::DescriptorImageInfo{
                .imageView = skyboxResources.radianceViews[0],
                .imageLayout = vk::ImageLayout::eGeneral,
            }};
        for (uint32_t i = 1; i < mipCount; ++i)
            imageInfos[i] = vk::DescriptorImageInfo{
                .imageView = skyboxResources.radianceViews[i],
                .imageLayout = vk::ImageLayout::eGeneral,
            };

        const StaticArray descriptorInfos{
            DescriptorInfo{skyboxResources.texture.imageInfo()},
            DescriptorInfo{imageInfos},
        };

        _prefilterRadiance.updateDescriptorSet(
            scopeAlloc.child_scope(), nextFrame, descriptorInfos);
        const vk::DescriptorSet storageSet =
            _prefilterRadiance.storageSet(nextFrame);

        skyboxResources.radiance.transition(cb, ImageState::ComputeShaderWrite);

        const auto _s = profiler->createCpuGpuScope(cb, "PrefilterRadiance");

        // TODO:
        // The number of groups is overkill here as each mip is a quarter of the
        // previous one. Most groups will early out.
        // Multiple tighter dispatches or a more complex group assignment in
        // shader?
        const uvec3 groups = uvec3{
            (uvec2{World::sSkyboxRadianceResolution} - 1u) / 16u + 1u,
            6 * skyboxResources.radiance.mipCount};

        _prefilterRadiance.record(
            cb,
            PrefilterRadiancePC{
                .mipCount = mipCount,
            },
            groups, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition
        // for all users
        skyboxResources.radiance.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }
    _generated = true;
}
