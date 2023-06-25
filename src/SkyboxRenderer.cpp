#include "SkyboxRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "RenderTargets.hpp"
#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

vk::Rect2D getRenderArea(
    const RenderResources &resources,
    const SkyboxRenderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.illumination).extent;
    assert(targetExtent.depth == 1);
    assert(
        targetExtent == resources.images.resource(inOutTargets.depth).extent);

    return vk::Rect2D{
        .offset = {0, 0},
        .extent =
            {
                targetExtent.width,
                targetExtent.height,
            },
    };
}

} // namespace

SkyboxRenderer::SkyboxRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);

    printf("Creating SkyboxRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("SkyboxRenderer shader compilation failed");

    createGraphicsPipelines(worldDSLayouts);
}

SkyboxRenderer::~SkyboxRenderer()
{
    if (_device != nullptr)
    {
        destroyGraphicsPipelines();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void SkyboxRenderer::recompileShaders(
    ScopedScratch scopeAlloc, const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(worldDSLayouts);
    }
}

void SkyboxRenderer::record(
    vk::CommandBuffer cb, const World &world, const RecordInOut &inOutTargets,
    const uint32_t nextFrame, Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

        recordBarriers(cb, inOutTargets);

        const Attachments attachments = createAttachments(inOutTargets);

        const auto _s = profiler->createCpuGpuScope(cb, "Skybox");

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachments.color,
            .pDepthAttachment = &attachments.depth,
        });

        // Skybox doesn't need to be drawn under opaque geometry but should be
        // before transparents
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            1, &world._skyboxDSs[nextFrame], 0, nullptr);

        setViewportScissor(cb, renderArea);

        world.drawSkybox(cb);

        cb.endRendering();
    }
}

bool SkyboxRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling SkyboxRenderer shaders\n");

    const Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.vert",
                                          .debugName = "skyboxVS",
                                      });
    const Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.frag",
                                          .debugName = "skyboxPS",
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _shaderStages = {
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
        };

        return true;
    }

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void SkyboxRenderer::recordBarriers(
    vk::CommandBuffer cb, const RecordInOut &inOutTargets) const
{
    const StaticArray barriers{
        _resources->images.transitionBarrier(
            inOutTargets.illumination,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
            }),
        _resources->images.transitionBarrier(
            inOutTargets.depth,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                .accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                .layout = vk::ImageLayout::eDepthAttachmentOptimal,
            }),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = asserted_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });
}

SkyboxRenderer::Attachments SkyboxRenderer::createAttachments(
    const RecordInOut &inOutTargets) const
{
    return Attachments{
        .color =
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.illumination).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
        .depth =
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.depth).view,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
    };
}

void SkyboxRenderer::destroyGraphicsPipelines()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void SkyboxRenderer::createGraphicsPipelines(
    const World::DSLayouts &worldDSLayouts)
{
    const vk::VertexInputBindingDescription vertexBindingDescription{
        .binding = 0,
        .stride = sizeof(vec3), // Only position
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const vk::VertexInputAttributeDescription vertexAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = 0,
    };
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexBindingDescription,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &vertexAttributeDescription,
    };

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &worldDSLayouts.skybox,
        });

    const vk::PipelineColorBlendAttachmentState blendAttachment =
        opaqueColorBlendAttachment();

    _pipeline = createGraphicsPipeline(
        _device->logical(), vk::PrimitiveTopology::eTriangleList,
        _pipelineLayout, vertInputInfo, vk::CullModeFlagBits::eNone,
        vk::CompareOp::eLessOrEqual, Span{&blendAttachment, 1}, _shaderStages,
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &sIlluminationFormat,
            .depthAttachmentFormat = sDepthFormat,
        },
        "SkyboxRenderer");
}
