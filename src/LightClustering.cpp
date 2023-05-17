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
    const vk::Extent2D &renderExtent, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    printf("Creating LightClustering\n");

    if (!compileShaders(scopeAlloc.child_scope()))
        throw std::runtime_error("LightClustering shader compilation failed");

    createDescriptorSets();

    _resources->buffers.lightClusters.indicesCount =
        _device->createTexelBuffer(TexelBufferCreateInfo{
            .bufferInfo =
                BufferCreateInfo{
                    .byteSize = sizeof(uint32_t),
                    .usage = vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                    .debugName = "LightClusteringIndicesCounter",
                },
            .format = vk::Format::eR32Uint,
            .supportAtomics = true,
        });

    recreate(renderExtent, camDSLayout, worldDSLayouts);
}

LightClustering::~LightClustering()
{
    if (_device != nullptr)
    {
        destroyViewportRelated();

        _device->destroy(_resources->buffers.lightClusters.indicesCount);
        _device->logical().destroy(
            _resources->buffers.lightClusters.descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
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

void LightClustering::recreate(
    const vk::Extent2D &renderExtent, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroyViewportRelated();

    createOutputs(renderExtent);
    updateDescriptorSets();
    createPipeline(camDSLayout, worldDSLayouts);
}

void LightClustering::record(
    vk::CommandBuffer cb, const Scene &scene, const Camera &cam,
    const vk::Rect2D &renderArea, const uint32_t nextFrame, Profiler *profiler)
{
    if (renderArea.offset != vk::Offset2D{})
        throw std::runtime_error("Offset area not implemented!");

    {
        const auto _s = profiler->createCpuGpuScope(cb, "LightClustering");

        const auto imageBarrier =
            _resources->buffers.lightClusters.pointers.transitionBarrier(
                ImageState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderWrite,
                    .layout = vk::ImageLayout::eGeneral,
                });

        const StaticArray bufferBarriers{
            _resources->buffers.lightClusters.indices.transitionBarrier(
                BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderWrite,
                }),
            _resources->buffers.lightClusters.indicesCount.transitionBarrier(
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

        cb.fillBuffer(
            _resources->buffers.lightClusters.indicesCount.handle, 0,
            _resources->buffers.lightClusters.indicesCount.size, 0);

        _resources->buffers.lightClusters.indicesCount.transition(
            cb, BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead |
                                  vk::AccessFlagBits2::eShaderWrite,
                });

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

        StaticArray<vk::DescriptorSet, 3> descriptorSets{VK_NULL_HANDLE};
        descriptorSets[sLightsBindingSet] =
            scene.lights.descriptorSets[nextFrame];
        descriptorSets[sLightClustersBindingSet] =
            _resources->buffers.lightClusters.descriptorSets[nextFrame];
        descriptorSets[sCameraBindingSet] = cam.descriptorSet(nextFrame);

        cb.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, _pipelineLayout,
            0, // firstSet
            asserted_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(), 0, nullptr);

        const ClusteringPCBlock pcBlock{
            .resolution =
                uvec2(renderArea.extent.width, renderArea.extent.height),
        };
        cb.pushConstants(
            _pipelineLayout, vk::ShaderStageFlagBits::eCompute,
            0, // offset
            sizeof(ClusteringPCBlock), &pcBlock);

        const auto &extent = _resources->buffers.lightClusters.pointers.extent;
        cb.dispatch(extent.width, extent.height, extent.depth);
    }
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

    const auto compSM = _device->compileShaderModule(
        scopeAlloc.child_scope(), Device::CompileShaderModuleArgs{
                                      .relPath = "shader/light_clustering.comp",
                                      .debugName = "lightClusteringCS",
                                      .defines = defines,
                                  });

    if (compSM.has_value())
    {
        _device->logical().destroy(_compSM);

        _compSM = *compSM;

        return true;
    }

    return false;
}

void LightClustering::destroyViewportRelated()
{
    if (_device != nullptr)
    {
        destroyPipeline();

        _device->destroy(_resources->buffers.lightClusters.pointers);
        _device->destroy(_resources->buffers.lightClusters.indices);
    }
}

void LightClustering::createOutputs(const vk::Extent2D &renderExtent)
{
    const auto pointersWidth = ((renderExtent.width - 1u) / clusterDim) + 1u;
    const auto pointersHeight = ((renderExtent.height - 1u) / clusterDim) + 1u;
    const auto pointersDepth = zSlices + 1;

    _resources->buffers.lightClusters.pointers =
        _device->createImage(ImageCreateInfo{
            .imageType = vk::ImageType::e3D,
            .format = vk::Format::eR32G32Uint,
            .width = pointersWidth,
            .height = pointersHeight,
            .depth = pointersDepth,
            .usageFlags = vk::ImageUsageFlagBits::eStorage,
            .debugName = "lightClusterPointers",
        });

    const vk::DeviceSize indicesSize =
        static_cast<vk::DeviceSize>(
            maxSpotIndicesPerTile + maxPointIndicesPerTile) *
        pointersWidth * pointersHeight * pointersDepth;
    _resources->buffers.lightClusters.indices =
        _device->createTexelBuffer(TexelBufferCreateInfo{
            .bufferInfo =
                BufferCreateInfo{
                    .byteSize = indicesSize * sizeof(uint16_t),
                    .usage = vk::BufferUsageFlagBits::eStorageTexelBuffer,
                    .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
                    .debugName = "lightClusterIndices",
                },
            .format = vk::Format::eR16Uint,
        });
}

void LightClustering::createDescriptorSets()
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
    _resources->buffers.lightClusters.descriptorSetLayout =
        _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });

    const StaticArray<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{
        _resources->buffers.lightClusters.descriptorSetLayout};
    _resources->staticDescriptorsAlloc.allocate(
        layouts, Span{
                     _resources->buffers.lightClusters.descriptorSets.data(),
                     _resources->buffers.lightClusters.descriptorSets.size()});
}

void LightClustering::updateDescriptorSets()
{
    const vk::DescriptorImageInfo pointersInfo{
        .imageView = _resources->buffers.lightClusters.pointers.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    StaticArray<vk::WriteDescriptorSet, MAX_FRAMES_IN_FLIGHT * 3>
        descriptorWrites;
    for (const auto &ds : _resources->buffers.lightClusters.descriptorSets)
    {
        descriptorWrites.push_back(vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &pointersInfo,
        });
        descriptorWrites.push_back(vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .pTexelBufferView =
                &_resources->buffers.lightClusters.indicesCount.view,
        });
        descriptorWrites.push_back(vk::WriteDescriptorSet{
            .dstSet = ds,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .pTexelBufferView = &_resources->buffers.lightClusters.indices.view,
        });
    }
    _device->logical().updateDescriptorSets(
        asserted_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}

void LightClustering::createPipeline(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    StaticArray<vk::DescriptorSetLayout, 3> setLayouts{VK_NULL_HANDLE};
    setLayouts[sLightsBindingSet] = worldDSLayouts.lights;
    setLayouts[sLightClustersBindingSet] =
        _resources->buffers.lightClusters.descriptorSetLayout;
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
