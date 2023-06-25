#include "LightClustering.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;
using namespace wheels;

namespace
{

constexpr uint32_t maxPointIndicesPerTile = 128;
constexpr uint32_t maxSpotIndicesPerTile = 128;

enum BindingSet : uint32_t
{
    LightsBindingSet = 0,
    LightClustersBindingSet = 1,
    CameraBindingSet = 2,
    BindingSetCount = 3,
};

struct ClusteringPCBlock
{
    uvec2 resolution;
};

} // namespace

LightClustering::LightClustering(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    assert(_device != nullptr);
    assert(_resources != nullptr);
    assert(staticDescriptorsAlloc != nullptr);

    printf("Creating LightClustering\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("LightClustering shader compilation failed");

    createDescriptorSets(scopeAlloc.child_scope(), staticDescriptorsAlloc);
    createPipeline(camDSLayout, worldDSLayouts);
}

LightClustering::~LightClustering()
{
    if (_device != nullptr)
    {
        destroyPipeline();

        _device->logical().destroy(_descriptorSetLayout);
        _device->logical().destroy(_compSM);
    }
}

[[nodiscard]] vk::DescriptorSetLayout LightClustering::descriptorSetLayout()
    const
{
    return _descriptorSetLayout;
}

void LightClustering::recompileShaders(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders(scopeAlloc.child_scope()))
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
    }
}

LightClustering::Output LightClustering::record(
    vk::CommandBuffer cb, const Scene &scene, const Camera &cam,
    const vk::Extent2D &renderExtent, const uint32_t nextFrame,
    Profiler *profiler)
{
    Output ret;
    {
        const auto _s = profiler->createCpuGpuScope(cb, "LightClustering");

        ret = createOutputs(renderExtent);
        updateDescriptorSet(nextFrame, ret);

        recordBarriers(cb, ret);

        { // Reset count
            const TexelBuffer &indicesCount =
                _resources->texelBuffers.resource(ret.indicesCount);

            cb.fillBuffer(indicesCount.handle, 0, indicesCount.size, 0);

            _resources->texelBuffers.transition(
                cb, ret.indicesCount,
                BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead |
                                  vk::AccessFlagBits2::eShaderWrite,
                });
        }

        { // Main dispatch
            cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

            StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
                VK_NULL_HANDLE};
            descriptorSets[LightsBindingSet] =
                scene.lights.descriptorSets[nextFrame];
            descriptorSets[LightClustersBindingSet] = ret.descriptorSet;
            descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);

            cb.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, _pipelineLayout,
                0, // firstSet
                asserted_cast<uint32_t>(descriptorSets.size()),
                descriptorSets.data(), 0, nullptr);

            const ClusteringPCBlock pcBlock{
                .resolution = uvec2(renderExtent.width, renderExtent.height),
            };
            cb.pushConstants(
                _pipelineLayout, vk::ShaderStageFlagBits::eCompute,
                0, // offset
                sizeof(ClusteringPCBlock), &pcBlock);

            const vk::Extent3D &extent =
                _resources->images.resource(ret.pointers).extent;
            cb.dispatch(extent.width, extent.height, extent.depth);
        }
    }

    return ret;
}

bool LightClustering::compileShaders(ScopedScratch scopeAlloc)
{
    printf("Compiling LightClustering shaders\n");

    String defines{scopeAlloc, 256};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    appendShaderDefines(defines);

    Optional<Device::ShaderCompileResult> compResult =
        _device->compileShaderModule(
            scopeAlloc.child_scope(),
            Device::CompileShaderModuleArgs{
                .relPath = "shader/light_clustering.comp",
                .debugName = "lightClusteringCS",
                .defines = defines,
            });

    if (compResult.has_value())
    {
        _device->logical().destroy(_compSM);

        ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(ClusteringPCBlock) == reflection.pushConstantsBytesize());

        _compSM = compResult->module;
        _shaderReflection = WHEELS_MOV(reflection);

        return true;
    }

    return false;
}

void LightClustering::recordBarriers(
    vk::CommandBuffer cb, const Output &output) const
{
    const vk::ImageMemoryBarrier2 imageBarrier =
        _resources->images.transitionBarrier(
            output.pointers,
            ImageState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
                .layout = vk::ImageLayout::eGeneral,
            });

    const StaticArray bufferBarriers{
        _resources->texelBuffers.transitionBarrier(
            output.indices,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .accessMask = vk::AccessFlagBits2::eShaderWrite,
            }),
        _resources->texelBuffers.transitionBarrier(
            output.indicesCount,
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                .accessMask = vk::AccessFlagBits2::eTransferWrite,
            }),
    };

    cb.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageBarrier,
    });
}

LightClustering::Output LightClustering::createOutputs(
    const vk::Extent2D &renderExtent)
{
    const auto pointersWidth = ((renderExtent.width - 1u) / clusterDim) + 1u;
    const auto pointersHeight = ((renderExtent.height - 1u) / clusterDim) + 1u;
    const auto pointersDepth = zSlices + 1;

    Output ret;

    ret.pointers = _resources->images.create(
        ImageDescription{
            .imageType = vk::ImageType::e3D,
            .format = vk::Format::eR32G32Uint,
            .width = pointersWidth,
            .height = pointersHeight,
            .depth = pointersDepth,
            .usageFlags = vk::ImageUsageFlagBits::eStorage,
        },
        "lightClusterPointers");

    ret.indicesCount = _resources->texelBuffers.create(
        TexelBufferDescription{
            .bufferDesc =
                BufferDescription{
                    .byteSize = sizeof(uint32_t),
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .format = vk::Format::eR32Uint,
            .supportAtomics = true,
        },
        "LightClusteringIndicesCounter");

    const vk::DeviceSize indicesSize =
        static_cast<vk::DeviceSize>(
            maxSpotIndicesPerTile + maxPointIndicesPerTile) *
        pointersWidth * pointersHeight * pointersDepth;

    ret.indices = _resources->texelBuffers.create(
        TexelBufferDescription{
            .bufferDesc =
                BufferDescription{
                    .byteSize = indicesSize * sizeof(uint16_t),
                    .usage = vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                },
            .format = vk::Format::eR16Uint,
        },
        "lightClusterIndices");

    return ret;
}

void LightClustering::createDescriptorSets(
    ScopedScratch scopeAlloc, DescriptorAllocator *staticDescriptorsAlloc)
{
    assert(_shaderReflection.has_value());
    const Array<vk::DescriptorSetLayoutBinding> layoutBindings =
        _shaderReflection->generateLayoutBindings(
            scopeAlloc, LightClustersBindingSet,
            vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eCompute);

    _descriptorSetLayout = _device->logical().createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
            .pBindings = layoutBindings.data(),
        });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _descriptorSetLayout};
    staticDescriptorsAlloc->allocate(
        layouts, Span{_descriptorSets.data(), _descriptorSets.size()});
}

void LightClustering::updateDescriptorSet(uint32_t nextFrame, Output &output)
{
    // TODO:
    // Don't update if resources are the same as before (for this DS index)?
    // Have to compare against both extent and previous native handle?

    const vk::DescriptorImageInfo pointersInfo{
        .imageView = _resources->images.resource(output.pointers).view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    const vk::DescriptorSet ds = _descriptorSets[nextFrame];

    assert(_shaderReflection.has_value());
    const StaticArray descriptorWrites =
        _shaderReflection->generateDescriptorWrites<3>(
            LightClustersBindingSet, ds,
            {
                Pair{0u, DescriptorInfoPtr{&pointersInfo}},
                Pair{
                    1u, DescriptorInfoPtr{&_resources->texelBuffers
                                               .resource(output.indicesCount)
                                               .view}},
                Pair{
                    2u, DescriptorInfoPtr{&_resources->texelBuffers
                                               .resource(output.indices)
                                               .view}},
            });

    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);

    output.descriptorSet = ds;
}

void LightClustering::createPipeline(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = worldDSLayouts.lights;
    setLayouts[LightClustersBindingSet] = _descriptorSetLayout;
    setLayouts[CameraBindingSet] = camDSLayout;

    const vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(ClusteringPCBlock),
    };
    _pipelineLayout =
        _device->logical().createPipelineLayout(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = asserted_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        });

    const vk::ComputePipelineCreateInfo createInfo{
        .stage =
            {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = _compSM,
                .pName = "main",
            },
        .layout = _pipelineLayout,
    };

    _pipeline = createComputePipeline(
        _device->logical(), createInfo, "LightClustering");
}

void LightClustering::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}
