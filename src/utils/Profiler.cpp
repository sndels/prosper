#include "Profiler.hpp"

#include "../gfx/Device.hpp"
#include "Utils.hpp"

using namespace wheels;

namespace
{

constexpr uint32_t sMaxScopeCount = 512;
// Each scope has a query for start and end
constexpr uint32_t sMaxTimestampCount = sMaxScopeCount * 2;
// TODO: Mesh shader stats
constexpr vk::QueryPipelineStatisticFlags sPipelineStatisticsFlags =
    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
    vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
constexpr size_t sStatTypeCount = asserted_cast<size_t>(
    std::popcount(static_cast<uint32_t>(sPipelineStatisticsFlags)));

} // namespace

GpuFrameProfiler::Scope::Scope(
    vk::CommandBuffer cb, QueryPools pools, const char *name,
    uint32_t queryIndex, bool includeStatistics)
: _cb{cb}
, _pools{pools}
, _queryIndex{queryIndex}
, _hasStatistics{includeStatistics}
{
    cb.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = name,
    });
    cb.writeTimestamp2(
        vk::PipelineStageFlagBits2::eTopOfPipe, _pools.timestamps,
        _queryIndex * 2);
    if (_hasStatistics)
        cb.beginQuery(_pools.statistics, _queryIndex, vk::QueryControlFlags{});
}

GpuFrameProfiler::Scope::~Scope()
{
    if (_cb)
    {
        if (_hasStatistics)
            _cb.endQuery(_pools.statistics, _queryIndex);
        _cb.writeTimestamp2(
            vk::PipelineStageFlagBits2::eBottomOfPipe, _pools.timestamps,
            _queryIndex * 2 + 1);
        _cb.endDebugUtilsLabelEXT();
    }
}

GpuFrameProfiler::Scope::Scope(GpuFrameProfiler::Scope &&other) noexcept
: _cb{other._cb}
, _pools{other._pools}
, _queryIndex{other._queryIndex}
, _hasStatistics{other._hasStatistics}
{
    other._cb = vk::CommandBuffer{};
}

GpuFrameProfiler::GpuFrameProfiler(wheels::Allocator &alloc, Device *device)
: _device{device}
, _timestampBuffer{device->createBuffer(BufferCreateInfo{
      .desc =
          BufferDescription{
              .byteSize = sizeof(uint64_t) * sMaxTimestampCount,
              .usage = vk::BufferUsageFlagBits::eTransferDst,
              .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent,
          },
      .debugName = "GpuProfilerTimestampReadback"})}
, _statisticsBuffer{device->createBuffer(BufferCreateInfo{
      .desc =
          BufferDescription{
              .byteSize = sizeof(uint32_t) * sStatTypeCount * sMaxScopeCount,
              .usage = vk::BufferUsageFlagBits::eTransferDst,
              .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent,
          },
      .debugName = "GpuProfilerStatisticsReadback"})}
, _queryScopeIndices{alloc, sMaxScopeCount}
, _scopeHasStats{alloc, sMaxScopeCount}
{
    _pools.timestamps =
        _device->logical().createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = sMaxTimestampCount});
    _pools.statistics =
        _device->logical().createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::ePipelineStatistics,
            .queryCount = sMaxScopeCount,
            .pipelineStatistics = sPipelineStatisticsFlags,
        });
}

GpuFrameProfiler::~GpuFrameProfiler()
{
    if (_device != nullptr)
    {
        _device->logical().destroyQueryPool(_pools.statistics);
        _device->logical().destroyQueryPool(_pools.timestamps);
        _device->destroy(_statisticsBuffer);
        _device->destroy(_timestampBuffer);
    }
}

GpuFrameProfiler::GpuFrameProfiler(GpuFrameProfiler &&other) noexcept
: _device{other._device}
, _timestampBuffer{WHEELS_MOV(other._timestampBuffer)}
, _statisticsBuffer{WHEELS_MOV(other._statisticsBuffer)}
, _pools{other._pools}
, _queryScopeIndices{WHEELS_MOV(other._queryScopeIndices)}
, _scopeHasStats{WHEELS_MOV(other._scopeHasStats)}
{
    other._device = nullptr;
}

GpuFrameProfiler &GpuFrameProfiler::operator=(GpuFrameProfiler &&other) noexcept
{
    if (this != &other)
    {
        _device = other._device;
        _timestampBuffer = WHEELS_MOV(other._timestampBuffer);
        _statisticsBuffer = WHEELS_MOV(other._statisticsBuffer);
        _pools = other._pools;
        _queryScopeIndices = WHEELS_MOV(other._queryScopeIndices);
        _scopeHasStats = WHEELS_MOV(other._scopeHasStats);

        other._device = nullptr;
    }
    return *this;
}

void GpuFrameProfiler::startFrame()
{
    // Might be more optimal to do this in a command buffer if we had some other
    // use that was ensured to happen before all other command buffers.
    _device->logical().resetQueryPool(_pools.timestamps, 0, sMaxTimestampCount);
    _device->logical().resetQueryPool(_pools.statistics, 0, sMaxScopeCount);
    _queryScopeIndices.clear();
    _scopeHasStats.clear();
}

void GpuFrameProfiler::endFrame(vk::CommandBuffer cb)
{
    cb.copyQueryPoolResults(
        _pools.timestamps, 0,
        asserted_cast<uint32_t>(_queryScopeIndices.size() * 2),
        _timestampBuffer.handle, 0, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    cb.copyQueryPoolResults(
        _pools.statistics, 0,
        asserted_cast<uint32_t>(_queryScopeIndices.size()),
        _statisticsBuffer.handle, 0, sizeof(uint32_t) * sStatTypeCount,
        vk::QueryResultFlags{});
}

GpuFrameProfiler::Scope GpuFrameProfiler::createScope(
    vk::CommandBuffer cb, const char *name, uint32_t index,
    bool includeStatistics)
{
    const auto queryIndex = asserted_cast<uint32_t>(_queryScopeIndices.size());
    _queryScopeIndices.push_back(index);
    _scopeHasStats.push_back(includeStatistics);
    return Scope{cb, _pools, name, queryIndex, includeStatistics};
}

Array<GpuFrameProfiler::ScopeData> GpuFrameProfiler::getData(Allocator &alloc)
{
    const auto timestampPeriodNanos = static_cast<double>(
        _device->properties().device.limits.timestampPeriod);

    // This is garbage if no frame has been completed with this profiler yet
    // Caller should make sure that isn't an issue.
    const auto *timestamps =
        reinterpret_cast<uint64_t *>(_timestampBuffer.mapped);
    const auto *stats = reinterpret_cast<uint32_t *>(_statisticsBuffer.mapped);

    const size_t scopeCount = _queryScopeIndices.size();
    WHEELS_ASSERT(scopeCount == _scopeHasStats.size());
    Array<ScopeData> ret{alloc, scopeCount};
    for (auto i = 0u; i < scopeCount; ++i)
    {
        // All bits valid should have been asserted on device creation
        const auto start = timestamps[static_cast<size_t>(i) * 2];
        const auto end = timestamps[static_cast<size_t>(i) * 2 + 1];
        const auto nanos = (end - start) * timestampPeriodNanos;
        const float millis = static_cast<float>(nanos * 1e-6);
        const bool hasStats = _scopeHasStats[i];

        ret.push_back(ScopeData{
            .index = _queryScopeIndices[i],
            .millis = millis,
            .stats = hasStats ? Optional{PipelineStatistics{
                                    .clipPrimitives =
                                        stats[static_cast<size_t>(i) * 5],
                                    .fragInvocations =
                                        stats[static_cast<size_t>(i) * 5 + 1],
                                }}
                              : Optional<PipelineStatistics>{},
        });
    };

    return ret;
}

CpuFrameProfiler::Scope::Scope(std::chrono::nanoseconds *output)
: _start{std::chrono::high_resolution_clock::now()}
, _output{output}
{
}

CpuFrameProfiler::Scope::~Scope()
{
    if (_output != nullptr)
        *_output = std::chrono::high_resolution_clock::now() - _start;
}

CpuFrameProfiler::Scope::Scope(CpuFrameProfiler::Scope &&other) noexcept
: _start{other._start}
, _output{other._output}
{
    other._output = nullptr;
}

CpuFrameProfiler::CpuFrameProfiler(wheels::Allocator &alloc)
: _nanos{alloc, sMaxScopeCount}
{
}

void CpuFrameProfiler::startFrame() { _nanos.clear(); }

[[nodiscard]] CpuFrameProfiler::Scope CpuFrameProfiler::createScope(
    uint32_t index)
{
    WHEELS_ASSERT(
        index == _nanos.size() &&
        "CpuFrameProfiler expects that all indices have a scope attached");
    (void)index;

    _nanos.emplace_back();

    return Scope{&_nanos.back()};
}

Array<CpuFrameProfiler::ScopeTime> CpuFrameProfiler::getTimes(Allocator &alloc)
{
    Array<ScopeTime> times{alloc, _nanos.size()};
    for (auto i = 0u; i < _nanos.size(); ++i)
    {
        times.push_back(ScopeTime{
            .index = i,
            .millis =
                std::chrono::duration<float, std::milli>(_nanos[i]).count(),
        });
    };

    return times;
}

Profiler::Profiler(Allocator &alloc, Device *device)
: _alloc{alloc}
, _cpuFrameProfiler{_alloc}
, _gpuFrameProfilers{_alloc, MAX_FRAMES_IN_FLIGHT}
, _currentFrameScopeNames{_alloc, sMaxScopeCount}
, _previousScopeNames{_alloc}
, _previousCpuScopeTimes{_alloc}
, _previousGpuScopeData{_alloc, sMaxScopeCount}
{
    for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _gpuFrameProfilers.emplace_back(_alloc, device);
        _previousScopeNames.emplace_back(_alloc, sMaxScopeCount);
        _previousCpuScopeTimes.emplace_back(_alloc, sMaxScopeCount);
    }
}

void Profiler::startCpuFrame()
{
    WHEELS_ASSERT(_debugState == DebugState::NewFrame);

    // Only clear transients for this profiling frame. We'll figure out which
    // frame's data we'll overwrite in endCpuFrame, when we know the gpu frame
    // index of this frame
    _currentFrameScopeNames.clear();

    _cpuFrameProfiler.startFrame();

    _debugState = DebugState::StartCpuCalled;
}

void Profiler::startGpuFrame(uint32_t frameIndex)
{
    WHEELS_ASSERT(_debugState == DebugState::StartCpuCalled);
    WHEELS_ASSERT(frameIndex < _gpuFrameProfilers.size());

    _currentFrame = frameIndex;

    // Store data from the previous iteration of this gpu frame index. We need
    // to read these before startFrame as that will reset the queries.
    _previousGpuScopeData = _gpuFrameProfilers[_currentFrame].getData(_alloc);

    _gpuFrameProfilers[_currentFrame].startFrame();

    _debugState = DebugState::StartGpuCalled;
}

void Profiler::endGpuFrame(vk::CommandBuffer cb)
{
    WHEELS_ASSERT(_debugState == DebugState::StartGpuCalled);

    _gpuFrameProfilers[_currentFrame].endFrame(cb);

    _debugState = DebugState::EndGpuCalled;
}

void Profiler::endCpuFrame()
{
    WHEELS_ASSERT(_debugState == DebugState::EndGpuCalled);
    WHEELS_ASSERT(_currentFrame < _previousScopeNames.size());
    WHEELS_ASSERT(_currentFrame < _previousCpuScopeTimes.size());

    // We now know which frame's data we gave out in getTimes() so let's
    // overwrite them
    _previousScopeNames[_currentFrame].clear();
    for (const auto &name : _currentFrameScopeNames)
        _previousScopeNames[_currentFrame].emplace_back(_alloc, name);
    _previousCpuScopeTimes[_currentFrame] = _cpuFrameProfiler.getTimes(_alloc);

    _debugState = DebugState::NewFrame;
}

Profiler::Scope Profiler::createCpuGpuScope(
    vk::CommandBuffer cb, const char *name, bool includeStatistics)
{
    WHEELS_ASSERT(_debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(_currentFrameScopeNames.size());
    WHEELS_ASSERT(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _currentFrameScopeNames.emplace_back(_alloc, name);

    return Scope{
        _gpuFrameProfilers[_currentFrame].createScope(
            cb, name, index, includeStatistics),
        _cpuFrameProfiler.createScope(index)};
}

Profiler::Scope Profiler::createCpuScope(const char *name)
{
    WHEELS_ASSERT(
        _debugState == DebugState::StartCpuCalled ||
        _debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(_currentFrameScopeNames.size());
    WHEELS_ASSERT(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _currentFrameScopeNames.emplace_back(_alloc, name);

    return Scope{_cpuFrameProfiler.createScope(index)};
}

Array<Profiler::ScopeData> Profiler::getPreviousData(Allocator &alloc)
{
    WHEELS_ASSERT(_debugState == DebugState::StartGpuCalled);

    const auto &scopeNames = _previousScopeNames[_currentFrame];
    // This also handles the first calls before any scopes are recorded for the
    // frame index and the gpu data is garbage. Rest of the calls should be with
    // valid data as we have waited for swap with the corresponding frame index
    if (scopeNames.empty())
        return {alloc};

    Array<ScopeData> ret{alloc, scopeNames.size()};
    for (const auto &n : scopeNames)
        ret.push_back(ScopeData{
            .name = n,
        });

    for (const auto &data : _previousGpuScopeData)
    {
        WHEELS_ASSERT(data.index < ret.size());
        ret[data.index].gpuMillis = data.millis;
        ret[data.index].gpuStats = data.stats;
    }

    for (const auto &t : _previousCpuScopeTimes[_currentFrame])
    {
        WHEELS_ASSERT(t.index < ret.size());
        ret[t.index].cpuMillis = t.millis;
    }

    return ret;
}
