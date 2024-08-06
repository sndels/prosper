#ifndef PROSPER_UTILS_PROFILER_HPP
#define PROSPER_UTILS_PROFILER_HPP

#include "../Allocators.hpp"
#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"
#include "../utils/Utils.hpp"

#include <chrono>
#include <wheels/assert.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

constexpr uint32_t sMaxScopeCount = 512;

struct PipelineStatistics
{
    uint32_t clipPrimitives{0};
    uint32_t fragInvocations{0};
};

class GpuFrameProfiler
{
  public:
    struct QueryPools
    {
        vk::QueryPool timestamps;
        vk::QueryPool statistics;
    };
    class Scope
    {
      public:
        ~Scope();

        Scope(Scope const &) = delete;
        Scope(Scope &&other) noexcept;
        Scope &operator=(Scope const &) = delete;
        // No assignment since it doesn't seem to make sense: new scope should
        // be a new object. Move ctor is enough to use in an optional.
        Scope &operator=(Scope &&) = delete;

      protected:
        Scope(
            vk::CommandBuffer cb, QueryPools pools, const char *name,
            uint32_t queryIndex, bool includeStatistics);

      private:
        vk::CommandBuffer m_cb;
        QueryPools m_pools;
        uint32_t m_queryIndex{0};
        bool m_hasStatistics;

        friend class GpuFrameProfiler;
    };

    struct ScopeData
    {
        uint32_t index{0xFFFF'FFFF};
        float millis{0.f};
        wheels::Optional<PipelineStatistics> stats;
    };

    GpuFrameProfiler() noexcept = default;
    ~GpuFrameProfiler();

    GpuFrameProfiler(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler(GpuFrameProfiler &&other) noexcept;
    GpuFrameProfiler &operator=(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler &operator=(GpuFrameProfiler &&other) noexcept;

  protected:
    void init();
    void destroy();
    void startFrame();
    void endFrame(vk::CommandBuffer cb);

    [[nodiscard]] Scope createScope(
        vk::CommandBuffer cb, const char *name, uint32_t index,
        bool includeStatistics);

    // This will read garbage if the corresponding frame index has yet to have
    // any frame complete.
    [[nodiscard]] wheels::Array<ScopeData> getData(wheels::Allocator &alloc);

  private:
    bool m_initialized{false};
    Buffer m_timestampBuffer;
    Buffer m_statisticsBuffer;
    QueryPools m_pools;
    wheels::Array<uint32_t> m_queryScopeIndices{
        gAllocators.general, sMaxScopeCount};
    wheels::Array<bool> m_scopeHasStats{gAllocators.general, sMaxScopeCount};

    friend class Profiler;
};

class CpuFrameProfiler
{
  public:
    class Scope
    {
      public:
        ~Scope();

        Scope(Scope const &) = delete;
        Scope(Scope &&other) noexcept;
        Scope &operator=(Scope const &other) = delete;
        // No assignment since it doesn't seem to make sense: new scope should
        // be a new object. Move ctor is enough to use in an optional.
        Scope &operator=(Scope &&) = delete;

      protected:
        Scope(std::chrono::nanoseconds *output);

      private:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
        std::chrono::nanoseconds *m_output;

        friend class CpuFrameProfiler;
    };

    struct ScopeTime
    {
        uint32_t index{0xFFFF'FFFF};
        float millis{0.f};
    };

    CpuFrameProfiler() noexcept = default;
    ~CpuFrameProfiler();

    void init();
    void destroy();

    CpuFrameProfiler(CpuFrameProfiler const &) = delete;
    CpuFrameProfiler(CpuFrameProfiler &&) = delete;
    CpuFrameProfiler &operator=(CpuFrameProfiler const &) = delete;
    CpuFrameProfiler &operator=(CpuFrameProfiler &&) = delete;

  protected:
    void startFrame();

    [[nodiscard]] Scope createScope(uint32_t index);

    [[nodiscard]] wheels::Array<ScopeTime> getTimes(wheels::Allocator &alloc);

  private:
    bool m_initialized{false};
    // Any non-trivially destructible members need to be cleaned up manually in
    // destroy(). Thus, calling the dtor on an already destroyed object needs to
    // also be supported for the member types.
    wheels::Array<uint32_t> m_queryScopeIndices{gAllocators.general};
    wheels::Array<std::chrono::nanoseconds> m_nanos{gAllocators.general};

    friend class Profiler;
};

class Profiler
{
  public:
    class Scope
    {
      public:
        ~Scope() = default;

        Scope(Scope const &) = delete;
        Scope(Scope &&other) = delete;
        Scope &operator=(Scope const &) = delete;
        Scope &operator=(Scope &&other) = delete;

      private:
        Scope(
            GpuFrameProfiler::Scope &&gpuScope,
            CpuFrameProfiler::Scope &&cpuScope)
        : m_gpuScope{WHEELS_MOV(gpuScope)}
        , m_cpuScope{WHEELS_MOV(cpuScope)}
        {
        }

        Scope(CpuFrameProfiler::Scope &&cpuScope)
        : m_cpuScope{WHEELS_MOV(cpuScope)}
        {
        }

        Scope(GpuFrameProfiler::Scope &&gpuScope)
        : m_gpuScope{WHEELS_MOV(gpuScope)}
        {
        }

        wheels::Optional<GpuFrameProfiler::Scope> m_gpuScope;
        wheels::Optional<CpuFrameProfiler::Scope> m_cpuScope;

        friend class Profiler;
    };

    struct ScopeData
    {
        // Name will be null-terminated at size()
        wheels::StrSpan name;
        float gpuMillis{-1.f};
        float cpuMillis{-1.f};
        wheels::Optional<PipelineStatistics> gpuStats;
    };

    Profiler() noexcept = default;
    ~Profiler();

    // Assume one profiler that is initialized in place
    Profiler(Profiler const &) = delete;
    Profiler(Profiler &&) = delete;
    Profiler &operator=(Profiler const &) = delete;
    Profiler &operator=(Profiler &&) = delete;

    void init();
    void destroy();

    // Should be called before startGpuFrame, whenever the cpu frame loop starts
    void startCpuFrame();
    // Should be called before any command buffer recording. Frame index is the
    // swapchain image index as that tells us which previous frame's profiling
    // data we use
    void startGpuFrame(uint32_t frameIndex);

    // Should be called with the frame's presenting cb after the present barrier
    // to piggyback gpu readback sync on it.
    // Note: All gpu scopes should end before the present barrier.
    void endGpuFrame(vk::CommandBuffer cb);
    // Should be called after endGpuFrame, whenever the cpu frame loop ends
    // Note: All cpu scopes should end before this call
    void endCpuFrame();

    // Scopes can be created between the startFrame and endFrame -calls
    [[nodiscard]] Scope createCpuScope(const char *name);

    // GPU scopes shouldn't contain barriers because it might produce weird
    // results when they block the current scope on work that belongs to the
    // previous one.
    [[nodiscard]] Scope createGpuScope(
        vk::CommandBuffer cb, const char *name, bool includeStatistics = false);

    // GPU scopes shouldn't contain barriers because it might produce weird
    // results when they block the current scope on work that belongs to the
    // previous one.
    [[nodiscard]] Scope createCpuGpuScope(
        vk::CommandBuffer cb, const char *name, bool includeStatistics = false);

    // Can be called after startGpuFrame to get the data from the last
    // iteration of the active frame index.
    [[nodiscard]] wheels::Array<Profiler::ScopeData> getPreviousData(
        wheels::Allocator &alloc);

  private:
    // Do validation of the calls as it's easy to do things in the wrong order
    enum class DebugState
    {
        NewFrame,
        StartCpuCalled,
        StartGpuCalled,
        EndGpuCalled,
    };

    bool m_initialized{false};
    // Any non-trivially destructible members need to be cleaned up manually in
    // destroy(). Thus, calling the dtor on an already destroyed object needs to
    // also be supported for the member types.
    DebugState m_debugState{DebugState::NewFrame};

    CpuFrameProfiler m_cpuFrameProfiler;
    wheels::Array<GpuFrameProfiler> m_gpuFrameProfilers{gAllocators.general};

    // There should be a 1:1 mapping between swap images and profiler frames so
    // that we know our gpu data has been filled when we read it back the next
    // time the same index comes up. We should also have 1:1 mapping between gpu
    // frames and the cpu frames that recorded them.
    uint32_t m_currentFrame{0};
    wheels::Array<wheels::String> m_currentFrameScopeNames{gAllocators.general};

    wheels::Array<wheels::Array<wheels::String>> m_previousScopeNames{
        gAllocators.general};
    wheels::Array<wheels::Array<CpuFrameProfiler::ScopeTime>>
        m_previousCpuScopeTimes{gAllocators.general};
    wheels::Array<GpuFrameProfiler::ScopeData> m_previousGpuScopeData{
        gAllocators.general};
};

// This is depended on by Device and init()/destroy() order relative to other
// similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern Profiler gProfiler;

// The scope variable is never accessed so let's reduce the noise with a macro
// zz* to push the local variable to the bottom of the locals list in debuggers
#define PROFILER_CPU_SCOPE(name)                                               \
    const Profiler::Scope TOKEN_APPEND(zzCpuScope, __LINE__) =                 \
        gProfiler.createCpuScope(name);

// The scope variable is never accessed so let's reduce the noise with a macro
// zz* to push the local variable to the bottom of the locals list in debuggers
#define PROFILER_GPU_SCOPE(cb, name)                                           \
    const Profiler::Scope TOKEN_APPEND(zzGpuScope, __LINE__) =                 \
        gProfiler.createGpuScope(cb, name, false);

// The scope variable is never accessed so let's reduce the noise with a macro
// zz* to push the local variable to the bottom of the locals list in debuggers
#define PROFILER_GPU_SCOPE_WITH_STATS(cb, name)                                \
    const Profiler::Scope TOKEN_APPEND(zzGpuScope, __LINE__) =                 \
        gProfiler.createGpuScope(cb, name, true);

// The scope variable is never accessed so let's reduce the noise with a macro
// zz* to push the local variable to the bottom of the locals list in debuggers
#define PROFILER_CPU_GPU_SCOPE(cb, name)                                       \
    const Profiler::Scope TOKEN_APPEND(zzCpuGpuScope, __LINE__) =              \
        gProfiler.createCpuGpuScope(cb, name, false);

// The scope variable is never accessed so let's reduce the noise with a macro
// zz* to push the local variable to the bottom of the locals list in debuggers
#define PROFILER_CPU_GPU_SCOPE_WITH_STATS(cb, name)                            \
    const Profiler::Scope TOKEN_APPEND(zzCpuGpuScope, __LINE__) =              \
        gProfiler.createCpuGpuScope(cb, name, true);

#endif // PROSPER_UTILS_PROFILER_HPP
