#include "DebugRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "../gfx/VkUtils.hpp"
#include "../utils/Utils.hpp"
#include "LightClustering.hpp"
#include "RenderTargets.hpp"

using namespace glm;
using namespace wheels;

namespace
{

enum BindingSet : uint32_t
{
    CameraBindingSet = 0,
    GeometryBuffersBindingSet = 1,
    BindingSetCount = 2,
};

vk::Rect2D getRenderArea(
    const RenderResources &resources,
    const DebugRenderer::RecordInOut &inOutTargets)
{
    const vk::Extent3D targetExtent =
        resources.images.resource(inOutTargets.color).extent;
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

DebugRenderer::DebugRenderer(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating DebugRenderer\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("DebugRenderer shader compilation failed");

    createBuffers();
    createDescriptorSets(staticDescriptorsAlloc);
    createGraphicsPipeline(camDSLayout);
}

DebugRenderer::~DebugRenderer()
{
    if (_device != nullptr)
    {
        _device->logical().destroy(_linesDSLayout);

        for (auto &ls : _resources->debugLines)
            _device->destroy(ls.buffer);

        destroyGraphicsPipeline();

        for (auto const &stage : _shaderStages)
            _device->logical().destroyShaderModule(stage.module);
    }
}

void DebugRenderer::recompileShaders(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyGraphicsPipeline();
        createGraphicsPipeline(camDSLayout);
    }
}

void DebugRenderer::record(
    vk::CommandBuffer cb, const Camera &cam, const RecordInOut &inOutTargets,
    const uint32_t nextFrame, Profiler *profiler) const
{
    assert(profiler != nullptr);

    {
        const vk::Rect2D renderArea = getRenderArea(*_resources, inOutTargets);

        recordBarriers(cb, inOutTargets);

        const Attachments attachments = createAttachments(inOutTargets);

        const auto _s = profiler->createCpuGpuScope(cb, "Debug", true);

        cb.beginRendering(vk::RenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachments.color,
            .pDepthAttachment = &attachments.depth,
        });

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
            VK_NULL_HANDLE};
        descriptorSets[CameraBindingSet] = cam.descriptorSet();
        descriptorSets[GeometryBuffersBindingSet] =
            _linesDescriptorSets[nextFrame];

        const uint32_t cameraOffset = cam.bufferOffset();

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 1, &cameraOffset);

        setViewportScissor(cb, renderArea);

        const auto &lines = _resources->debugLines[nextFrame];
        // No need for lines barrier, writes are mapped

        cb.draw(lines.count * 2, 1, 0, 0);

        cb.endRendering();
    }
}

bool DebugRenderer::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling DebugRenderer shaders\n");

    String vertDefines{scopeAlloc, 128};
    appendDefineStr(vertDefines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(vertDefines, "GEOMETRY_SET", GeometryBuffersBindingSet);

    const Optional<Device::ShaderCompileResult> vertResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_lines.vert",
                                          .debugName = "debugLinesVS",
                                          .defines = vertDefines,
                                      });

    const Optional<Device::ShaderCompileResult> fragResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                          .relPath = "shader/debug_color.frag",
                                          .debugName = "debugColorPS",
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
            }};

        return true;
    }

    if (vertResult.has_value())
        _device->logical().destroy(vertResult->module);
    if (fragResult.has_value())
        _device->logical().destroy(fragResult->module);

    return false;
}

void DebugRenderer::recordBarriers(
    vk::CommandBuffer cb, const RecordInOut &inOutTargets) const
{
    transition<2>(
        *_resources, cb,
        {
            {inOutTargets.color, ImageState::ColorAttachmentWrite},
            {inOutTargets.depth, ImageState::DepthAttachmentReadWrite},
        });
}

DebugRenderer::Attachments DebugRenderer::createAttachments(
    const RecordInOut &inOutTargets) const
{
    return Attachments{
        .color =
            vk::RenderingAttachmentInfo{
                .imageView =
                    _resources->images.resource(inOutTargets.color).view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            },
        .depth = vk::RenderingAttachmentInfo{
            .imageView = _resources->images.resource(inOutTargets.depth).view,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        }};
}

void DebugRenderer::destroyGraphicsPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}

void DebugRenderer::createBuffers()
{
    for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
        _resources->debugLines.push_back(DebugLines{
            .buffer = _device->createBuffer(BufferCreateInfo{
                .desc =
                    BufferDescription{
                        .byteSize =
                            DebugLines::sMaxLineCount * DebugLines::sLineBytes,
                        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                        .properties =
                            vk::MemoryPropertyFlagBits::eHostCoherent |
                            vk::MemoryPropertyFlagBits::eHostVisible,
                    },
                .createMapped = true,
                .debugName = "DebugLines",
            }),
        });
}

void DebugRenderer::createDescriptorSets(
    DescriptorAllocator *staticDescriptorsAlloc)
{
    const vk::DescriptorSetLayoutBinding layoutBinding{
        .binding = 0, // binding
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1, // descriptorCount
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };
    _linesDSLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &layoutBinding,
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _linesDSLayout};
    _linesDescriptorSets.resize(
        _linesDescriptorSets.capacity(), VK_NULL_HANDLE);
    staticDescriptorsAlloc->allocate(layouts, _linesDescriptorSets);

    for (size_t i = 0; i < _linesDescriptorSets.size(); ++i)
    {
        const vk::DescriptorBufferInfo info{
            .buffer = _resources->debugLines[i].buffer.handle,
            .range = VK_WHOLE_SIZE,
        };

        const vk::WriteDescriptorSet descriptorWrite{
            .dstSet = _linesDescriptorSets[i],
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &info,
        };
        _device->logical().updateDescriptorSets(
            1, &descriptorWrite, 0, nullptr);
    }
}

void DebugRenderer::createGraphicsPipeline(
    const vk::DescriptorSetLayout camDSLayout)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[CameraBindingSet] = camDSLayout;
    setLayouts[GeometryBuffersBindingSet] = _linesDSLayout;

    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        });

    const vk::PipelineColorBlendAttachmentState blendAttachment =
        opaqueColorBlendAttachment();

    // Empty as we'll load vertices manually from a buffer
    const vk::PipelineVertexInputStateCreateInfo vertInputInfo;

    _pipeline = ::createGraphicsPipeline(
        _device->logical(), vk::PrimitiveTopology::eLineList, _pipelineLayout,
        vertInputInfo, vk::CullModeFlagBits::eBack, vk::CompareOp::eGreater,
        Span{&blendAttachment, 1}, _shaderStages,
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &sIlluminationFormat,
            .depthAttachmentFormat = sDepthFormat,
        },
        "DebugRenderer::Lines");
}
