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
    CameraBindingSet = 1,
    LightClustersBindingSet = 2,
    BindingSetCount = 3,
};

struct ClusteringPCBlock
{
    uvec2 resolution;
};

ComputePass::Shader shaderDefinitionCallback(Allocator &alloc)
{
    String defines{alloc, 256};
    appendDefineStr(defines, "LIGHTS_SET", LightsBindingSet);
    appendDefineStr(defines, "CAMERA_SET", CameraBindingSet);
    appendDefineStr(defines, "LIGHT_CLUSTERS_SET", LightClustersBindingSet);
    PointLights::appendShaderDefines(defines);
    SpotLights::appendShaderDefines(defines);
    LightClustering::appendShaderDefines(defines);

    return ComputePass::Shader{
        .relPath = "shader/light_clustering.comp",
        .debugName = String{alloc, "LightClusteringCS"},
        .defines = WHEELS_MOV(defines),
    };
}

StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> externalDsLayouts(
    const vk::DescriptorSetLayout &camDSLayout,
    const World::DSLayouts &worldDSLayout)
{
    StaticArray<vk::DescriptorSetLayout, BindingSetCount - 1> setLayouts{
        VK_NULL_HANDLE};
    setLayouts[LightsBindingSet] = worldDSLayout.lights;
    setLayouts[CameraBindingSet] = camDSLayout;
    return setLayouts;
}

} // namespace

LightClustering::LightClustering(
    ScopedScratch scopeAlloc, Device *device, RenderResources *resources,
    DescriptorAllocator *staticDescriptorsAlloc,
    const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
: _resources{resources}
, _computePass{
      WHEELS_MOV(scopeAlloc),
      device,
      staticDescriptorsAlloc,
      shaderDefinitionCallback,
      LightClustersBindingSet,
      externalDsLayouts(camDSLayout, worldDSLayouts),
      vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment}
{
    assert(_resources != nullptr);
}

vk::DescriptorSetLayout LightClustering::descriptorSetLayout() const
{
    return _computePass.storageSetLayout();
}

void LightClustering::recompileShaders(
    ScopedScratch scopeAlloc, const vk::DescriptorSetLayout camDSLayout,
    const World::DSLayouts &worldDSLayouts)
{
    _computePass.recompileShader(
        WHEELS_MOV(scopeAlloc), shaderDefinitionCallback,
        externalDsLayouts(camDSLayout, worldDSLayouts));
}

LightClustering::Output LightClustering::record(
    vk::CommandBuffer cb, const Scene &scene, const Camera &cam,
    const vk::Extent2D &renderExtent, const uint32_t nextFrame,
    Profiler *profiler)
{
    Output ret;
    {
        ret = createOutputs(renderExtent);

        _computePass.updateDescriptorSet(
            nextFrame,
            StaticArray{
                DescriptorInfo{vk::DescriptorImageInfo{
                    .imageView = _resources->images.resource(ret.pointers).view,
                    .imageLayout = vk::ImageLayout::eGeneral,
                }},
                DescriptorInfo{
                    _resources->texelBuffers.resource(ret.indicesCount).view},
                DescriptorInfo{
                    _resources->texelBuffers.resource(ret.indices).view},
            });
        ret.descriptorSet = _computePass.storageSet(nextFrame);

        transition<1, 2>(
            *_resources, cb,
            {
                {ret.pointers, ImageState::ComputeShaderWrite},
            },
            {
                {ret.indices, BufferState::ComputeShaderWrite},
                {ret.indicesCount, BufferState::TransferDst},
            });

        const auto _s = profiler->createCpuGpuScope(cb, "LightClustering");

        { // Reset count
            const TexelBuffer &indicesCount =
                _resources->texelBuffers.resource(ret.indicesCount);

            cb.fillBuffer(indicesCount.handle, 0, indicesCount.size, 0);

            _resources->texelBuffers.transition(
                cb, ret.indicesCount, BufferState::ComputeShaderReadWrite);
        }

        { // Main dispatch
            const ClusteringPCBlock pcBlock{
                .resolution = uvec2(renderExtent.width, renderExtent.height),
            };

            StaticArray<vk::DescriptorSet, BindingSetCount> descriptorSets{
                VK_NULL_HANDLE};
            descriptorSets[LightsBindingSet] =
                scene.lights.descriptorSets[nextFrame];
            descriptorSets[CameraBindingSet] = cam.descriptorSet(nextFrame);
            descriptorSets[LightClustersBindingSet] = ret.descriptorSet;

            const vk::Extent3D &extent =
                _resources->images.resource(ret.pointers).extent;
            const uvec3 groups{extent.width, extent.height, extent.depth};

            _computePass.record(cb, pcBlock, groups, descriptorSets);
        }
    }

    return ret;
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
            .usageFlags = vk::ImageUsageFlagBits::eSampled | // Debug
                          vk::ImageUsageFlagBits::eStorage,
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
