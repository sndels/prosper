#include "ImageBasedLighting.hpp"

#include "scene/World.hpp"
#include "scene/WorldRenderStructs.hpp"
#include "utils/Profiler.hpp"

#include <glm/glm.hpp>
#include <shader_structs/push_constants/prefilter_radiance.h>

using namespace wheels;
using namespace glm;

namespace
{

ComputePass::Shader sampleIrradianceShaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(
        defines, "OUT_RESOLUTION",
        SkyboxResources::sSkyboxIrradianceResolution);
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
        defines, "OUT_RESOLUTION", SkyboxResources::sSpecularBrdfLutResolution);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/ibl/integrate_specular_brdf.comp",
        .debugName = String{alloc, "IntegrateSpecularBrdfCS"},
        .defines = WHEELS_MOV(defines),
    };
}

ComputePass::Shader prefilterRadianceShaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 32;
    String defines{alloc, len};
    appendDefineStr(
        defines, "OUT_RESOLUTION", SkyboxResources::sSkyboxRadianceResolution);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/ibl/prefilter_radiance.comp",
        .debugName = String{alloc, "PrefilterRadianceCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

void ImageBasedLighting::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_sampleIrradiance.init(
        scopeAlloc.child_scope(), sampleIrradianceShaderDefinitionCallback);
    m_integrateSpecularBrdf.init(
        scopeAlloc.child_scope(),
        integrateSpecularBrdfShaderDefinitionCallback);
    m_prefilterRadiance.init(
        scopeAlloc.child_scope(), prefilterRadianceShaderDefinitionCallback);

    m_initialized = true;
}

bool ImageBasedLighting::isGenerated() const
{
    WHEELS_ASSERT(m_initialized);

    return m_generated;
}

void ImageBasedLighting::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    m_sampleIrradiance.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        sampleIrradianceShaderDefinitionCallback);
    m_integrateSpecularBrdf.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        integrateSpecularBrdfShaderDefinitionCallback);
    m_prefilterRadiance.recompileShader(
        scopeAlloc.child_scope(), changedFiles,
        prefilterRadianceShaderDefinitionCallback);
}

void ImageBasedLighting::recordGeneration(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, World &world,
    uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    SkyboxResources &skyboxResources = world.skyboxResources();

    {
        PROFILER_CPU_SCOPE("SampleIrradiance");

        const StaticArray descriptorInfos{{
            DescriptorInfo{skyboxResources.texture.imageInfo()},
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = skyboxResources.irradiance.view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        }};
        const vk::DescriptorSet storageSet =
            m_sampleIrradiance.updateStorageSet(
                scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        skyboxResources.irradiance.transition(
            cb, ImageState::ComputeShaderWrite);

        PROFILER_GPU_SCOPE(cb, "SampleIrradiance");

        const uvec3 groupCount = m_sampleIrradiance.groupCount(
            uvec3{uvec2{SkyboxResources::sSkyboxIrradianceResolution}, 6u});

        m_sampleIrradiance.record(cb, groupCount, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition
        // for all users
        skyboxResources.irradiance.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }

    {
        PROFILER_CPU_SCOPE("IntegrateSpecularBrdf");

        const StaticArray descriptorInfos{
            DescriptorInfo{vk::DescriptorImageInfo{
                .imageView = skyboxResources.specularBrdfLut.view,
                .imageLayout = vk::ImageLayout::eGeneral,
            }},
        };
        const vk::DescriptorSet storageSet =
            m_integrateSpecularBrdf.updateStorageSet(
                scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        skyboxResources.specularBrdfLut.transition(
            cb, ImageState::ComputeShaderWrite);

        PROFILER_GPU_SCOPE(cb, "IntegrateSpecularBrdf");

        const uvec3 groupCount = m_integrateSpecularBrdf.groupCount(
            uvec3{uvec2{SkyboxResources::sSpecularBrdfLutResolution}, 1u});

        m_integrateSpecularBrdf.record(cb, groupCount, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition for
        // all users
        skyboxResources.specularBrdfLut.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }

    {
        PROFILER_CPU_SCOPE("PrefilterRadiance");

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

        const StaticArray descriptorInfos{{
            DescriptorInfo{skyboxResources.texture.imageInfo()},
            DescriptorInfo{imageInfos},
        }};

        const vk::DescriptorSet storageSet =
            m_prefilterRadiance.updateStorageSet(
                scopeAlloc.child_scope(), nextFrame, descriptorInfos);

        skyboxResources.radiance.transition(cb, ImageState::ComputeShaderWrite);

        PROFILER_GPU_SCOPE(cb, "PrefilterRadiance");

        // TODO:
        // The number of groups is overkill here as each mip is a quarter of the
        // previous one. Most groups will early out.
        // Multiple tighter dispatches or a more complex group assignment in
        // shader?
        const uvec3 groupCount = m_prefilterRadiance.groupCount(uvec3{
            uvec2{SkyboxResources::sSkyboxRadianceResolution},
            6 * skyboxResources.radiance.mipCount});

        m_prefilterRadiance.record(
            cb,
            PrefilterRadiancePC{
                .mipCount = mipCount,
            },
            groupCount, Span{&storageSet, 1});

        // Transition so that the texture can be bound without transition
        // for all users
        skyboxResources.radiance.transition(
            cb, ImageState::ComputeShaderSampledRead |
                    ImageState::FragmentShaderSampledRead |
                    ImageState::RayTracingSampledRead);
    }
    m_generated = true;
}
