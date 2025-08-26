#include "Compose.hpp"

#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"

using namespace glm;
using namespace wheels;

namespace render::rtdi
{

namespace
{

enum BindingSet : uint8_t
{
    StorageBindingSet,
    BindingSetCount,
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    const size_t len = 128;
    String defines{alloc, len};
    appendDefineStr(defines, "STORAGE_SET", StorageBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    return ComputePass::Shader{
        .relPath = "shader/restir_di/compose.comp",
        .debugName = String{alloc, "RtDiComposeCS"},
        .defines = WHEELS_MOV(defines),
    };
}

} // namespace

void Compose::init(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(!m_initialized);

    m_computePass.init(
        WHEELS_MOV(scopeAlloc),
        [](Allocator &alloc) { return shaderDefinitionCallback(alloc); },
        ComputePassOptions{
            .storageSetIndex = StorageBindingSet,
        });

    m_initialized = true;
}

bool Compose::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles)
{
    WHEELS_ASSERT(m_initialized);

    return m_computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), changedFiles,
        [](Allocator &alloc) { return shaderDefinitionCallback(alloc); });
}

Compose::Output Compose::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const Input &input,
    const uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Compose");

    Output ret;
    {
        const vk::Extent2D renderExtent =
            getExtent2D(input.diffuseIllumination);

        ret.illumination = createIllumination(renderExtent, "RtDiCompose");

        const vk::DescriptorSet storageSet = m_computePass.updateStorageSet(
            scopeAlloc.child_scope(), nextFrame,
            StaticArray{{
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.diffuseIllumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = gRenderResources.images
                                     ->resource(input.specularIllumination)
                                     .view,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView =
                        gRenderResources.images->resource(ret.illumination)
                            .view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                gfx::DescriptorInfo{vk::DescriptorImageInfo{
                    .sampler = gRenderResources.nearestSampler,
                }},
            }});

        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 3>{{
                    {input.diffuseIllumination,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {input.specularIllumination,
                     gfx::ImageState::ComputeShaderSampledRead},
                    {ret.illumination, gfx::ImageState::ComputeShaderWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Compose");

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[StorageBindingSet] = storageSet;

        const uvec3 groupCount = m_computePass.groupCount(
            glm::uvec3{renderExtent.width, renderExtent.height, 1u});

        m_computePass.record(cb, groupCount, descriptorSets);
    }

    return ret;
}

} // namespace render::rtdi
