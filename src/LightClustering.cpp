#include "LightClustering.hpp"

// CMake doesn't seem to support MSVC /external -stuff yet
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

#include <glm/gtc/matrix_transform.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include "Utils.hpp"
#include "VkUtils.hpp"

using namespace glm;

namespace
{
// These have to match the shader
const uint32_t clusterDim = 32;
const uint32_t zSlices = 16;
const uint32_t maxPointIndicesPerTile = 128;
const uint32_t maxSpotIndicesPerTile = 128;

struct ClusteringPCBlock
{
    uvec2 resolution;
};

} // namespace

LightClustering::LightClustering(
    Device *device, RenderResources *resources,
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _device{device}
, _resources{resources}
{
    fprintf(stderr, "Creating LightClustering\n");

    if (!compileShaders())
        throw std::runtime_error("LightClustering shader compilation failed");

    const std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageTexelBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment |
                          vk::ShaderStageFlagBits::eCompute,
        },
    }};
    _resources->buffers.lightClusters.descriptorSetLayout =
        _device->logical().createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = asserted_cast<uint32_t>(layoutBindings.size()),
                .pBindings = layoutBindings.data(),
            });

    _resources->buffers.lightClusters.indicesCount = _device->createTexelBuffer(
        "LightClusteringIndicesCounter", vk::Format::eR32Uint, sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eStorageTexelBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, true);

    recreateSwapchainRelated(swapConfig, camDSLayout, worldDSLayouts);
}

LightClustering::~LightClustering()
{
    if (_device != nullptr)
    {
        destroySwapchainRelated();

        _device->destroy(_resources->buffers.lightClusters.indicesCount);
        _device->logical().destroy(
            _resources->buffers.lightClusters.descriptorSetLayout);

        _device->logical().destroy(_compSM);
    }
}

void LightClustering::recompileShaders(
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    if (compileShaders())
    {
        destroyPipeline();
        createPipeline(camDSLayout, worldDSLayouts);
    }
}

void LightClustering::recreateSwapchainRelated(
    const SwapchainConfig &swapConfig,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    destroySwapchainRelated();

    createOutputs(swapConfig);
    createDescriptorSets(swapConfig);
    createPipeline(camDSLayout, worldDSLayouts);
    createCommandBuffers(swapConfig);
}

vk::CommandBuffer LightClustering::recordCommandBuffer(
    const Scene &scene, const Camera &cam, const vk::Rect2D &renderArea,
    const uint32_t nextImage)
{
    if (renderArea.offset != vk::Offset2D{})
        throw std::runtime_error("Offset area not implemented!");

    const auto buffer = _commandBuffers[nextImage];
    buffer.reset();

    buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    buffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = "LightClustering",
    });

    const auto imageBarrier =
        _resources->buffers.lightClusters.pointers.transitionBarrier(ImageState{
            .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .accessMask = vk::AccessFlagBits2::eShaderWrite,
            .layout = vk::ImageLayout::eGeneral,
        });

    const std::array<vk::BufferMemoryBarrier2, 2> bufferBarriers{
        _resources->buffers.lightClusters.indices.transitionBarrier(BufferState{
            .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .accessMask = vk::AccessFlagBits2::eShaderWrite,
        }),
        _resources->buffers.lightClusters.indicesCount.transitionBarrier(
            BufferState{
                .stageMask = vk::PipelineStageFlagBits2::eTransfer,
                .accessMask = vk::AccessFlagBits2::eTransferWrite,
            }),
    };

    buffer.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount =
            asserted_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageBarrier,
    });

    buffer.fillBuffer(
        _resources->buffers.lightClusters.indicesCount.handle, 0,
        _resources->buffers.lightClusters.indicesCount.size, 0);

    _resources->buffers.lightClusters.indicesCount.transition(
        buffer, BufferState{
                    .stageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .accessMask = vk::AccessFlagBits2::eShaderRead |
                                  vk::AccessFlagBits2::eShaderWrite,
                });

    buffer.bindPipeline(vk::PipelineBindPoint::eCompute, _pipeline);

    const std::array<vk::DescriptorSet, 3> descriptorSets{
        scene.lights.descriptorSets[nextImage],
        cam.descriptorSet(nextImage),
        _resources->buffers.lightClusters.descriptorSets[nextImage],
    };
    buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, _pipelineLayout,
        0, // firstSet
        asserted_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
        0, nullptr);

    const ClusteringPCBlock pcBlock{
        .resolution = uvec2(renderArea.extent.width, renderArea.extent.height),
    };
    buffer.pushConstants(
        _pipelineLayout, vk::ShaderStageFlagBits::eCompute,
        0, // offset
        sizeof(ClusteringPCBlock), &pcBlock);

    const auto &extent = _resources->buffers.lightClusters.pointers.extent;
    buffer.dispatch(extent.width, extent.height, extent.depth);

    buffer.endDebugUtilsLabelEXT(); // LightClustering

    buffer.end();

    return buffer;
}

bool LightClustering::compileShaders()
{
    fprintf(stderr, "Compiling LightClustering shaders\n");

    const auto compSM = _device->compileShaderModule(
        "shader/light_clustering.comp", "lightClusteringCS");

    if (compSM)
    {
        _device->logical().destroy(_compSM);

        _compSM = *compSM;

        return true;
    }

    return false;
}

void LightClustering::destroySwapchainRelated()
{
    if (_device != nullptr)
    {
        if (!_commandBuffers.empty())
        {
            _device->logical().freeCommandBuffers(
                _device->graphicsPool(),
                asserted_cast<uint32_t>(_commandBuffers.size()),
                _commandBuffers.data());
        }

        destroyPipeline();

        _device->destroy(_resources->buffers.lightClusters.pointers);
        _device->destroy(_resources->buffers.lightClusters.indices);
    }
}

void LightClustering::createOutputs(const SwapchainConfig &swapConfig)
{
    const vk::ImageSubresourceRange subresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    const vk::Extent3D pointersExtent{
        .width = ((swapConfig.extent.width - 1u) / clusterDim) + 1u,
        .height = ((swapConfig.extent.height - 1u) / clusterDim) + 1u,
        .depth = zSlices + 1,
    };
    _resources->buffers.lightClusters.pointers = _device->createImage(
        "lightClusterPointers", vk::ImageType::e3D, pointersExtent,
        vk::Format::eR32G32Uint, subresourceRange, vk::ImageViewType::e3D,
        vk::ImageTiling::eOptimal, vk::ImageCreateFlagBits{},
        vk::ImageUsageFlagBits::eStorage,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    const vk::DeviceSize indicesSize =
        static_cast<vk::DeviceSize>(
            maxSpotIndicesPerTile + maxPointIndicesPerTile) *
        pointersExtent.width * pointersExtent.height;
    _resources->buffers.lightClusters.indices = _device->createTexelBuffer(
        "lightClusterIndices", vk::Format::eR16Uint,
        indicesSize * sizeof(uint16_t),
        vk::BufferUsageFlagBits::eStorageTexelBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, false);
}

void LightClustering::createDescriptorSets(const SwapchainConfig &swapConfig)
{
    const std::vector<vk::DescriptorSetLayout> layouts(
        swapConfig.imageCount,
        _resources->buffers.lightClusters.descriptorSetLayout);
    _resources->buffers.lightClusters.descriptorSets =
        _device->logical().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = _resources->descriptorPools.swapchainRelated,
            .descriptorSetCount = asserted_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        });

    vk::DescriptorImageInfo pointersInfo{
        .imageView = _resources->buffers.lightClusters.pointers.view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };
    std::vector<vk::WriteDescriptorSet> descriptorWrites;
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
    const std::array<vk::DescriptorSetLayout, 3> setLayouts{
        worldDSLayouts.lights,
        camDSLayout,
        _resources->buffers.lightClusters.descriptorSetLayout,
    };
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

void LightClustering::createCommandBuffers(const SwapchainConfig &swapConfig)
{
    _commandBuffers =
        _device->logical().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = _device->graphicsPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = swapConfig.imageCount,
        });
}
