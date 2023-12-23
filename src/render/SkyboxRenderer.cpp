#include "SkyboxRenderer.hpp"

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

enum BindingSet : uint32_t
{
    SkyboxBindingSet = 0,
    CameraBindingSet,
    BindingSetCount,
};

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

} // namespace

SkyboxRenderer::SkyboxRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    const vk::DescriptorSetLayout camDSLayout,
    const WorldDSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    WHEELS_ASSERT(_device != nullptr);
    WHEELS_ASSERT(_resources != nullptr);

    printf("Creating SkyboxRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("SkyboxRenderer shader compilation failed");

    createGraphicsPipelines(camDSLayout, worldDSLayouts);
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
    const vk::DescriptorSetLayout camDSLayout,
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
        createGraphicsPipelines(camDSLayout, worldDSLayouts);
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
            .colorAttachmentCount =
                asserted_cast<uint32_t>(attachments.color.size()),
            .pColorAttachments = attachments.color.data(),
            .pDepthAttachment = &attachments.depth,
        });

        // Skybox doesn't need to be drawn under opaque geometry but should be
        // before transparents
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        const WorldDescriptorSets &worldDSes = world.descriptorSets();

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[SkyboxBindingSet] = worldDSes.skybox;
        descriptorSets[CameraBindingSet] = cam.descriptorSet();

        const uint32_t camOffset = cam.bufferOffset();

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &camOffset);

        setViewportScissor(cb, renderArea);

        world.drawSkybox(cb);

        cb.endRendering();
    }
}

bool SkyboxRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling SkyboxRenderer shaders\n");

    const size_t len = 48;
    String defines{scopeAlloc, len};
    appendDefineStr(defines, "SKYBOX_SET", SkyboxBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    WHEELS_ASSERT(defines.size() <= len);

    Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/skybox.vert",
                                          .debugName = "skyboxVS",
                                          .defines = defines,
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
        _fragReflection = WHEELS_MOV(fragResult->reflection);

        _shaderStages = {{
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
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void SkyboxRenderer::recordBarriers(
    vk::CommandBuffer cb, const RecordInOut &inOutTargets) const
{
    transition<3>(
        *_resources, cb,
        {{
            {inOutTargets.illumination, ImageState::ColorAttachmentWrite},
            {inOutTargets.velocity, ImageState::ColorAttachmentWrite},
            {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
        }});
}

SkyboxRenderer::Attachments SkyboxRenderer::createAttachments(
    const RecordInOut &inOutTargets) const
{
    return Attachments{
        .color = {{
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.illumination).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.velocity).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
        }},
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
    const vk::DescriptorSetLayout camDSLayout,
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

    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[SkyboxBindingSet] = worldDSLayouts.skybox;
    setLayouts[CameraBindingSet] = camDSLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    const StaticArray colorAttachmentFormats{{
        sIlluminationFormat,
        sVelocityFormat,
    }};

    const StaticArray<vk::PipelineColorBlendAttachmentState, 2>
        colorBlendAttachments{opaqueColorBlendAttachment()};

    _pipeline = createGraphicsPipeline(
        _device->logical(),
        GraphicsPipelineInfo{
            .layout = _pipelineLayout,
            .vertInputInfo = vertInputInfo,
            .colorBlendAttachments = colorBlendAttachments,
            .shaderStages = _shaderStages,
            .renderingInfo =
                vk::PipelineRenderingCreateInfo{
                    .colorAttachmentCount = asserted_cast<uint32_t>(
                        colorAttachmentFormats.capacity()),
                    .pColorAttachmentFormats = colorAttachmentFormats.data(),
                    .depthAttachmentFormat = sDepthFormat,
                },
            .cullMode = vk::CullModeFlagBits::eNone,
            .depthCompareOp = vk::CompareOp::eGreaterOrEqual,
            .debugName = "SkyboxRenderer",
        });
}
