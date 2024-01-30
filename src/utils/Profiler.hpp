#ifndef PROSPER_UTILS_PROFILER_HPP
#define PROSPER_UTILS_PROFILER_HPP

#include "../gfx/Fwd.hpp"
#include "../gfx/Resources.hpp"

#include <wheels/allocators/allocator.hpp>
#include <wheels/containers/array.hpp>
#include <wheels/containers/optional.hpp>
#include <wheels/containers/span.hpp>
#include <wheels/containers/string.hpp>

#include <chrono>

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
        vk::CommandBuffer _cb;
        QueryPools _pools;
        uint32_t _queryIndex{0};
        bool _hasStatistics;

        friend class GpuFrameProfiler;
    };

    struct ScopeData
    {
        uint32_t index{0xFFFFFFFF};
        float millis{0.f};
        wheels::Optional<PipelineStatistics> stats;
    };

    GpuFrameProfiler(wheels::Allocator &alloc) noexcept;
    ~GpuFrameProfiler();

    GpuFrameProfiler(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler(GpuFrameProfiler &&other) noexcept;
    GpuFrameProfiler &operator=(GpuFrameProfiler const &) = delete;
    GpuFrameProfiler &operator=(GpuFrameProfiler &&other) noexcept;

  protected:
    void init(Device *device);
    void startFrame();
    void endFrame(vk::CommandBuffer cb);

    [[nodiscard]] Scope createScope(
        vk::CommandBuffer cb, const char *name, uint32_t index,
        bool includeStatistics);

    // This will read garbage if the corresponding frame index has yet to have
    // any frame complete.
    [[nodiscard]] wheels::Array<ScopeData> getData(wheels::Allocator &alloc);

  private:
    Device *_device{nullptr};
    Buffer _timestampBuffer;
    Buffer _statisticsBuffer;
    QueryPools _pools;
    wheels::Array<uint32_t> _queryScopeIndices;
    wheels::Array<bool> _scopeHasStats;

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
        std::chrono::time_point<std::chrono::high_resolution_clock> _start;
        std::chrono::nanoseconds *_output;

        friend class CpuFrameProfiler;
    };

    struct ScopeTime
    {
        uint32_t index{0xFFFFFFFF};
        float millis{0.f};
    };

    CpuFrameProfiler(wheels::Allocator &alloc) noexcept;
    ~CpuFrameProfiler() = default;

    CpuFrameProfiler(CpuFrameProfiler const &) = delete;
    CpuFrameProfiler(CpuFrameProfiler &&) = delete;
    CpuFrameProfiler &operator=(CpuFrameProfiler const &) = delete;
    CpuFrameProfiler &operator=(CpuFrameProfiler &&) = delete;

  protected:
    void startFrame();

    [[nodiscard]] Scope createScope(uint32_t index);

    [[nodiscard]] wheels::Array<ScopeTime> getTimes(wheels::Allocator &alloc);

  private:
    wheels::Array<std::chrono::nanoseconds> _nanos;

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
        : _gpuScope{WHEELS_MOV(gpuScope)}
        , _cpuScope{WHEELS_MOV(cpuScope)}
        {
        }

        Scope(CpuFrameProfiler::Scope &&cpuScope)
        : _cpuScope{WHEELS_MOV(cpuScope)}
        {
        }

        wheels::Optional<GpuFrameProfiler::Scope> _gpuScope;
        wheels::Optional<CpuFrameProfiler::Scope> _cpuScope;

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

    Profiler(wheels::Allocator &alloc) noexcept;
    ~Profiler() = default;

    // Assume one profiler that is initialized in place
    Profiler(Profiler const &) = delete;
    Profiler(Profiler &&) = delete;
    Profiler &operator=(Profiler const &) = delete;
    Profiler &operator=(Profiler &&) = delete;

    void init(Device *device);

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

    bool _initialized{false};
    DebugState _debugState{DebugState::NewFrame};

    wheels::Allocator &_alloc;

    CpuFrameProfiler _cpuFrameProfiler;
    wheels::Array<GpuFrameProfiler> _gpuFrameProfilers;

    // There should be a 1:1 mapping between swap images and profiler frames so
    // that we know our gpu data has been filled when we read it back the next
    // time the same index comes up. We should also have 1:1 mapping between gpu
    // frames and the cpu frames that recorded them.
    uint32_t _currentFrame{0};
    wheels::Array<wheels::String> _currentFrameScopeNames;

    wheels::Array<wheels::Array<wheels::String>> _previousScopeNames;
    wheels::Array<wheels::Array<CpuFrameProfiler::ScopeTime>>
        _previousCpuScopeTimes;
    wheels::Array<GpuFrameProfiler::ScopeData> _previousGpuScopeData;
};

#endif // PROSPER_UTILS_PROFILER_HPP
