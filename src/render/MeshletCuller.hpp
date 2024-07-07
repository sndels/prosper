#ifndef PROSPER_RENDER_MESHLET_CULLER_HPP
#define PROSPER_RENDER_MESHLET_CULLER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../scene/Fwd.hpp"
#include "../utils/Fwd.hpp"
#include "../utils/Utils.hpp"
#include "ComputePass.hpp"
#include "Fwd.hpp"
#include "RenderResourceHandle.hpp"

#include <wheels/allocators/scoped_scratch.hpp>
#include <wheels/containers/hash_set.hpp>
#include <wheels/containers/inline_array.hpp>
#include <wheels/containers/static_array.hpp>

struct MeshletCullerOutput
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
        wheels::ScopedScratch scopeAlloc, const WorldDSLayouts &worldDsLayouts,
        vk::DescriptorSetLayout camDsLayout);

    void recompileShaders(
        wheels::ScopedScratch scopeAlloc,
        const wheels::HashSet<std::filesystem::path> &changedFiles,
        const WorldDSLayouts &WorldDSLayouts,
        vk::DescriptorSetLayout camDsLayout);

    void startFrame();

    enum class Mode
    {
        Opaque,
        Transparent,
    };
    [[nodiscard]] MeshletCullerOutput record(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
        const World &world, const Camera &cam, uint32_t nextFrame,
        const char *debugPrefix, DrawStats *drawStats);

  private:
    [[nodiscard]] BufferHandle recordGenerateList(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb, Mode mode,
        const World &world, uint32_t nextFrame, const char *debugPrefix,
        DrawStats *drawStats);

    [[nodiscard]] BufferHandle recordWriteCullerArgs(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        uint32_t nextFrame, BufferHandle drawList, const char *debugPrefix);

    struct CullerInput
    {
        BufferHandle dataBuffer;
        BufferHandle argumentBuffer;
    };
    [[nodiscard]] MeshletCullerOutput recordCullList(
        wheels::ScopedScratch scopeAlloc, vk::CommandBuffer cb,
        const World &world, const Camera &cam, uint32_t nextFrame,
        const CullerInput &input, const char *debugPrefix);

    bool m_initialized{false};

    ComputePass m_drawListGenerator;
    ComputePass m_cullerArgumentsWriter;
    ComputePass m_drawListCuller;
};

#endif // PROSPER_RENDER_MESHLET_CULLER_HPP
