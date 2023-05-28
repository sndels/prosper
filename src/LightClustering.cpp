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

constexpr uint32_t sLightsBindingSet = 0;
constexpr uint32_t sLightClustersBindingSet = 1;
constexpr uint32_t sCameraBindingSet = 2;

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

    createDescriptorSets(staticDescriptorsAlloc);
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

            StaticArray<vk::DescriptorSet, 3> descriptorSets{VK_NULL_HANDLE};
            descriptorSets[sLightsBindingSet] =
                scene.lights.descriptorSets[nextFrame];
            descriptorSets[sLightClustersBindingSet] = ret.descriptorSet;
            descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextFrame);

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
    appendDefineStr(defines, "LIGHTS_SET", sLightsBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", sLightClustersBindingSet);
    appendDefineStr(defines, "CAMERA_SET", sCameraBindingSet);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    appendShaderDefines(defines);

    const Optional<Device::ShaderCompileResult> compResult =
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

        const ShaderReflection &reflection = compResult->reflection;
        assert(sizeof(ClusteringPCBlock) == reflection.pushConstantsBytesize());

        _compSM = compResult->module;

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
    DescriptorAllocator *staticDescriptorsAlloc)
{
    const StaticArray layoutBindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
    };
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
    const StaticArray descriptorWrites{
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &pointersInfo,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .pTexelBufferView =
                &_resources->texelBuffers.resource(output.indicesCount).view,
        },
        vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .pTexelBufferView =
                &_resources->texelBuffers.resource(output.indices).view,
        }};
    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);

    output.descriptorSet = ds;
}

void LightClustering::createPipeline(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, 3> setLayouts{VK_NULL_HANDLE};
    setLayouts[sLightsBindingSet] = worldDSLayouts.lights;
    setLayouts[sLightClustersBindingSet] = _descriptorSetLayout;
    setLayouts[sCameraBindingSet] = camDSLayout;

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

    {
        auto pipeline = _device->logical().createComputePipeline(
            vk::PipelineCache{}, createInfo);
        if (pipeline.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create pbr pipeline");

        _pipeline = pipeline.value;

        _device->logical().setDebugUtilsObjectNameEXT(
            vk::DebugUtilsObjectNameInfoEXT{
                .objectType = vk::ObjectType::ePipeline,
                .objectHandle = reinterpret_cast<uint64_t>(
                    static_cast<VkPipeline>(_pipeline)),
                .pObjectName = "LightClustering",
            });
    }
}

void LightClustering::destroyPipeline()
{
    _device->logical().destroy(_pipeline);
    _device->logical().destroy(_pipelineLayout);
}
