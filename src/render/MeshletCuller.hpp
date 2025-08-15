#ifndef PROSPER_RENDER_MESHLET_CULLER_HPP
#define PROSPER_RENDER_MESHLET_CULLER_HPP

#include "render/ComputePass.hpp"
#include "render/Fwd.hpp"
#include "render/RenderResourceHandle.hpp"
#include "scene/Fwd.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/static_array.hpp>

namespace render
{

struct MeshletCullerFirstPhaseOutput
{
    BufferHandle dataBuffer;
    BufferHandle argumentBuffer;
    wheels::Optional<BufferHandle> secondPhaseInput;
};

struct MeshletCullerSecondPhaseOutput
{
    BufferHandle dataBuffer;
    BufferHandle argumentBuffer;
};

class MeshletCuller
{
  public:
    MeshletCuller() noexcept = default;
    ~MeshletCuller() = default;

    MeshletCuller(const MeshletCuller &other) = delete;
    MeshletCuller(MeshletCuller &&other) = delete;
    MeshletCuller &operator=(const MeshletCuller &other) = delete;
    MeshletCuller &operator=(MeshletCuller &&other) = delete;

    void init(
        wheels::ScopedScratch scopeAlloc,
        const scene::WorldDSLayouts &worldDsLayouts,
        vk::DescriptorSetLayout camDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const scene::WorldDSLayouts &WorldDSLayouts,
        vk::DescriptorSetLayout camDsLayout);

    void startFrame();

    enum class Mode : uint8_t
    {
        Opaque,
        Transparent,
    };
    // Creates and culls meshlet draw lists from full scene data. Also creates
    // input for second phase if inHierarchicalDepth is given.
    [[nodiscard]] MeshletCullerFirstPhaseOutput recordFirstPhase(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
        const scene::World &world, const scene::Camera &cam, uint32_t nextFrame,
        const wheels::Optional<ImageHandle> &inHierarchicalDepth,
        wheels::StrSpan debugPrefix, DrawStats &drawStats);

    // Culls the input draw lists. Intended to use with depth drawn with the
    // first pass outputs
    [[nodiscard]] MeshletCullerSecondPhaseOutput recordSecondPhase(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::World &world, const scene::Camera &cam, uint32_t nextFrame,
        BufferHandle inputBuffer, ImageHandle inHierarchicalDepth,
        wheels::StrSpan debugPrefix);

  private:
    [[nodiscard]] BufferHandle recordGenerateList(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
        const scene::World &world, uint32_t nextFrame,
        wheels::StrSpan debugPrefix, DrawStats &drawStats);

    [[nodiscard]] BufferHandle recordWriteCullerArgs(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, BufferHandle drawList, wheels::StrSpan debugPrefix);

    struct CullInput
    {
        BufferHandle dataBuffer;
        BufferHandle argumentBuffer;
        wheels::Optional<ImageHandle> hierarchicalDepth;
    };
    struct CullOutput
    {
        BufferHandle dataBuffer;
        BufferHandle argumentBuffer;
        wheels::Optional<BufferHandle> secondPhaseInput;
    };
    [[nodiscard]] CullOutput recordCullList(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const scene::World &world, const scene::Camera &cam, uint32_t nextFrame,
        const CullInput &input, bool outputSecondPhaseInputs,
        wheels::StrSpan debugPrefix);

    bool m_initialized{false};

    ComputePass m_drawListGenerator;
    ComputePass m_cullerArgumentsWriter;
    ComputePass m_drawListCuller;
};

} // namespace render

#endif // PROSPER_RENDER_MESHLET_CULLER_HPP
