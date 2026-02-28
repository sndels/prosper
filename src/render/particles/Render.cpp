#include "Render.hpp"

#include "gfx/DescriptorAllocator.hpp"
#include "gfx/Device.hpp"
#include "gfx/Resources.hpp"
#include "gfx/VkUtils.hpp"
#include "render/RenderResources.hpp"
#include "render/RenderTargets.hpp"
#include "render/Utils.hpp"
#include "render/particles/Particles.hpp"
#include "scene/Camera.hpp"
#include "utils/Logger.hpp"
#include "utils/Profiler.hpp"
#include "utils/Utils.hpp"
#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/glm.hpp>
#include <shader_structs/push_constants/particles/render.h>

using namespace glm;
using namespace wheels;

namespace render::particles
{

namespace
{

enum BindingSet : uint8_t
{
    CameraSet,
    ParticlesSet,
    BindingSetCount,
};
}

Render::~Render()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    destroyGraphicsPipelines();

    gfx::gDevice.logical().destroy(m_setLayout);

    for (auto const &stage : m_shaderStages)
        gfx::gDevice.logical().destroyShaderModule(stage.module);
}

void Render::init(
    wheels::ScopedScratch scopeAlloc, vk::DescriptorSetLayout cameraDSLayout)
{
    WHEELS_ASSERT(!m_initialized);

    LOG_INFO("Creating Particles::Render");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("Particles::Render shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope());
    createGraphicsPipelines(cameraDSLayout);

    m_initialized = true;
}

void Render::recompileShaders(
    wheels::ScopedScratch scopeAlloc,
    const wheels::HashSet<std::filesystem::path> &changedFiles,
    vk::DescriptorSetLayout cameraDSLayout)
{
    WHEELS_ASSERT(m_initialized);

    WHEELS_ASSERT(m_vertReflection.has_value());
    WHEELS_ASSERT(m_fragReflection.has_value());
    if (!m_vertReflection->affected(changedFiles) &&
        !m_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(cameraDSLayout);
    }
}

void Render::record(
    ScopedScratch scopeAlloc, vk::CommandBuffer cb, const scene::Camera &cam,
    const InputOutput &inOut, uint32_t nextFrame)
{
    WHEELS_ASSERT(m_initialized);

    PROFILER_CPU_SCOPE("  Render::Particles");

    // TODO:
    // Match period to TAA period instead of cycling through unique Bayer matrix
    // offsets?
    m_frameIndex = (m_frameIndex + 1) % 64;

    {
        const vk::Rect2D renderArea = getRect2D(inOut.inOutIllumination);

        const vk::DescriptorSet ds = m_descriptorSets[nextFrame];

        updateDescriptorSet(scopeAlloc.child_scope(), ds, inOut);

        inOut.inParticles.transition(cb, gfx::BufferState::VertexShaderRead);
        transition(
            WHEELS_MOV(scopeAlloc), cb,
            Transitions{
                .images = StaticArray<ImageTransition, 2>{{
                    {inOut.inOutIllumination,
                     gfx::ImageState::ColorAttachmentReadWrite},
                    {inOut.inOutDepth,
                     gfx::ImageState::DepthAttachmentReadWrite},
                }},
            });

        PROFILER_GPU_SCOPE(cb, "  Render::Particles");

        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView =
                gRenderResources.images->resource(inOut.inOutIllumination).view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView =
                gRenderResources.images->resource(inOut.inOutDepth).view,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment,
            .pDepthAttachment = &depthAttachment,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

        const StaticArray descriptorSets = {{cam.descriptorSet(), ds}};

        const uint32_t camOffset = cam.bufferOffset();
        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0,
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &camOffset);

        gfx::setViewportScissor(cb, renderArea);

        const RenderPC pcBlock{
            .frameIndex = m_frameIndex,
        };
        cb.pushConstants(
            m_pipelineLayout, vk::ShaderStageFlagBits::eFragment,
            0, // offset
            sizeof(pcBlock), &pcBlock);

        cb.draw(4, Particles::sMaxParticleCount, 0, 0);

        cb.endRendering();
    }
}

bool Render::compileShaders(ScopedScratch scopeAlloc)
{
    const size_t vertDefsLen = 48;
    String vertDefines{scopeAlloc, vertDefsLen};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraSet);
    appendDefineStr(vertDefines, "PARTICLES_SET", ParticlesSet);
    WHEELS_ASSERT(vertDefines.size() <= vertDefsLen);

    Optional<gfx::Device::ShaderCompileResult> vertResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(),
            gfx::Device::CompileShaderModuleArgs{
                .relPath = "shader/particles/render.vert",
                .debugName = "particlesVS",
                .defines = vertDefines,
            });

    Optional<gfx::Device::ShaderCompileResult> fragResult =
        gfx::gDevice.compileShaderModule(
            scopeAlloc.child_scope(),
            gfx::Device::CompileShaderModuleArgs{
                .relPath = "shader/particles/render.frag",
                .debugName = "particlesPS",
            });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : m_shaderStages)
            gfx::gDevice.logical().destroyShaderModule(stage.module);

        m_vertReflection = WHEELS_MOV(vertResult->reflection);
        m_fragReflection = WHEELS_MOV(fragResult->reflection);

        m_shaderStages = {{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertResult->module,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragResult->module,
                .pName = "main",
            },
        }};

        return true;
    }

    if (vertResult.has_value())
        gfx::gDevice.logical().destroy(vertResult->module);
    if (fragResult.has_value())
        gfx::gDevice.logical().destroy(fragResult->module);

    return false;
}

void Render::createDescriptorSets(ScopedScratch scopeAlloc)
{
    WHEELS_ASSERT(m_vertReflection.has_value());
    m_setLayout = m_vertReflection->createDescriptorSetLayout(
        WHEELS_MOV(scopeAlloc), ParticlesSet, vk::ShaderStageFlagBits::eVertex);

    const StaticArray<vk::DescriptorSetLayout, 2> layouts{m_setLayout};
    const StaticArray<const char *, 2> debugNames{"ParticlesRender"};
    gfx::gStaticDescriptorsAlloc.allocate(
        layouts, debugNames, m_descriptorSets.mut_span());
}

void Render::updateDescriptorSet(
    ScopedScratch scopeAlloc, vk::DescriptorSet ds,
    const InputOutput &inOut) const
{
    const StaticArray infos{{
        gfx::DescriptorInfo{vk::DescriptorBufferInfo{
            .buffer = inOut.inParticles.handle,
            .range = VK_WHOLE_SIZE,
        }},
    }};

    WHEELS_ASSERT(m_vertReflection.has_value());
    const wheels::Array descriptorWrites =
        m_vertReflection->generateDescriptorWrites(
            scopeAlloc, ParticlesSet, ds, infos);

    gfx::gDevice.logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void Render::destroyGraphicsPipelines()
{
    gfx::gDevice.logical().destroy(m_pipeline);
    gfx::gDevice.logical().destroy(m_pipelineLayout);
}

void Render::createGraphicsPipelines(vk::DescriptorSetLayout cameraDSLayout)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraSet] = cameraDSLayout;
    setLayouts[ParticlesSet] = m_setLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(RenderPC),
    };
    m_pipelineLayout = gfx::gDevice.logical().createPipelineLayout(
        vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    {
        const StaticArray<vk::PipelineColorBlendAttachmentState, 1>
            colorBlendAttachments{gfx::opaqueColorBlendAttachment()};

        // Empty as we'll load vertices manually from a buffer
        const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

        m_pipeline = gfx::createGraphicsPipeline(
            gfx::gDevice.logical(),
            gfx::GraphicsPipelineInfo{
                .layout = m_pipelineLayout,
                .vertInputInfo = &vertInputInfo,
                .colorBlendAttachments = colorBlendAttachments,
                .shaderStages = m_shaderStages,
                .renderingInfo =
                    vk::PipelineRenderingCreateInfo{
                        .colorAttachmentCount = 1,
                        .pColorAttachmentFormats = &sIlluminationFormat,
                        .depthAttachmentFormat = sDepthFormat,
                    },
                .topology = vk::PrimitiveTopology::eTriangleStrip,
                .debugName = "Particles::Render",
            });
    }
}

} // namespace render::particles
