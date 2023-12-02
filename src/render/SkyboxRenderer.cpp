#include "SkyboxRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "../gfx/Device.hpp"
#include "../gfx/VkUtils.hpp"
#include "../scene/Camera.hpp"
#include "../scene/World.hpp"
#include "../utils/Profiler.hpp"
#include "../utils/Utils.hpp"
#include "RenderResources.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t sSkyboxBindingSet = 0;

vk::Rect2D getRenderArea(
    const RenderResources &resources,
    const SkyboxRenderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.illumination).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);
    WHEELS_ASSERT(
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

struct PCBlock
{
    mat4 worldToClip;
};

} // namespace

SkyboxRenderer::SkyboxRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const WorldDSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);

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
    ScopedScratch scopeAlloc,
    const HashSet<std::filesystem::path> &changedFiles,
    const WorldDSLayouts &worldDSLayouts)
{
    WHEELS_ASSERT(_vertReflection.has_value());
    WHEELS_ASSERT(_fragReflection.has_value());
    if (!_vertReflection->affected(changedFiles) &&
        !_fragReflection->affected(changedFiles))
        return;

    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipelines();
        createGraphicsPipelines(worldDSLayouts);
    }
}

void SkyboxRenderer::record(
    vk::CommandBuffer cb, const World &world, const Camera &cam,
    const RecordInOut &inOutTargets, Profiler *profiler) const
{
    WHEELS_ASSERT(profiler != nullptr);

    {
        const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

        recordBarriers(cb, inOutTargets);

        const Attachments attachments = createAttachments(inOutTargets);

        const auto _s = profiler->createCpuGpuScope(cb, "Skybox", true);

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
            1, &world._skyboxDS, 0, nullptr);

        setViewportScissor(cb, renderArea);

        const PCBlock pcBlock{
            .worldToClip = cam.cameraToClip() * mat4(mat3(cam.worldToCamera())),
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eVertex,
            0, // offset
            sizeof(PCBlock), &pcBlock);

        world.drawSkybox(cb);

        cb.endRendering();
    }
}

bool SkyboxRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling SkyboxRenderer shaders\n");

    const size_t len = 32;
    String defines{scopeAlloc, len};
    appendDefineStr(defines, "SKYBOX_SET", sSkyboxBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.vert",
                                          .debugName = "skyboxVS",
                                      });
    Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.frag",
                                          .debugName = "skyboxPS",
                                          .defines = defines,
                                      });

    if (vertResult.has_value() && fragResult.has_value())
    {
        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);

        _vertReflection = WHEELS_MOV(vertResult->reflection);
        WHEELS_ASSERT(
            sizeof(PCBlock) == _vertReflection->pushConstantsBytesize());

        _fragReflection = WHEELS_MOV(fragResult->reflection);

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
    transition<2>(
        *_resources, cb,
        {
            {inOutTargets.illumination, ImageState::ColorAttachmentWrite},
            {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
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
    const WorldDSLayouts &worldDSLayouts)
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
    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(PCBlock),
    };

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &worldDSLayouts.skybox,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::PipelineColorBlendAttachmentState blendAttachment =
        opaqueColorBlendAttachment();

    _pipeline = createGraphicsPipeline(
        _device->logical(), vk::PrimitiveTopology::eTriangleList,
        _pipelineLayout, vertInputInfo, vk::CullModeFlagBits::eNone,
        vk::CompareOp::eGreaterOrEqual, Span{&blendAttachment, 1},
        _shaderStages,
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &sIlluminationFormat,
            .depthAttachmentFormat = sDepthFormat,
        },
        "SkyboxRenderer");
}
