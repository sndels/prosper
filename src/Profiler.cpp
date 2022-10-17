#include "Profiler.hpp"

#include "Utils.hpp"

namespace
{

constexpr uint32_t sMaxScopeCount = 512;
// Each scope has a query for start and end
constexpr uint32_t sMaxQueryCount = sMaxScopeCount * 2;

} // namespace

GpuFrameProfiler::Scope::Scope(
    vk::CommandBuffer cb, vk::QueryPool queryPool, const std::string &name,
    uint32_t queryIndex)
: _cb{cb}
, _queryPool{queryPool}
, _queryIndex{queryIndex}
{
    cb.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = name.c_str(),
    });
    cb.writeTimestamp2(
        vk::PipelineStageFlagBits2::eTopOfPipe, _queryPool, _queryIndex * 2);
}

GpuFrameProfiler::Scope::~Scope()
{
    if (_cb)
    {
        _cb.writeTimestamp2(
            vk::PipelineStageFlagBits2::eBottomOfPipe, _queryPool,
            _queryIndex * 2 + 1);
        _cb.endDebugUtilsLabelEXT();
    }
}

GpuFrameProfiler::Scope::Scope(GpuFrameProfiler::Scope &&other) noexcept
: _cb{other._cb}
, _queryPool{other._queryPool}
, _queryIndex{other._queryIndex}
{
    other._cb = vk::CommandBuffer{};
}

GpuFrameProfiler::Scope &GpuFrameProfiler::Scope::operator=(
    GpuFrameProfiler::Scope &&other) noexcept
{
    if (this != &other)
    {
        _cb = other._cb;
        _queryPool = other._queryPool;
        _queryIndex = other._queryIndex;

        other._cb = vk::CommandBuffer{};
    }
    return *this;
}

GpuFrameProfiler::GpuFrameProfiler(Device *device)
: _device{device}
, _buffer{device->createBuffer(BufferCreateInfo{
      .byteSize = sizeof(uint64_t) * sMaxQueryCount,
      .usage = vk::BufferUsageFlagBits::eTransferDst,
      .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
      .createMapped = true,
      .debugName = "GpuProfilerReadback"})}
{
    _queryScopeIndices.reserve(sMaxScopeCount);
    _queryPool = _device->logical().createQueryPool(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp, .queryCount = sMaxQueryCount});
}

GpuFrameProfiler::~GpuFrameProfiler()
{
    if (_device != nullptr)
    {
        _device->logical().destroyQueryPool(_queryPool);
        _device->destroy(_buffer);
    }
}

GpuFrameProfiler::GpuFrameProfiler(GpuFrameProfiler &&other) noexcept
: _device{other._device}
, _buffer{other._buffer}
, _queryPool{other._queryPool}
, _queryScopeIndices{std::move(other._queryScopeIndices)}
{
    other._device = nullptr;
}

GpuFrameProfiler &GpuFrameProfiler::operator=(GpuFrameProfiler &&other) noexcept
{
    if (this != &other)
    {
        _device = other._device;
        _buffer = other._buffer;
        _queryPool = other._queryPool;
        _queryScopeIndices = std::move(other._queryScopeIndices);

        other._device = nullptr;
    }
    return *this;
}

void GpuFrameProfiler::startFrame()
{
    // Might be more optimal to do this in a command buffer if we had some other
    // use that was ensured to happen before all other command buffers.
    _device->logical().resetQueryPool(_queryPool, 0, sMaxQueryCount);
    _queryScopeIndices.clear();
}

void GpuFrameProfiler::endFrame(vk::CommandBuffer cb)
{
    cb.copyQueryPoolResults(
        _queryPool, 0, asserted_cast<uint32_t>(_queryScopeIndices.size() * 2),
        _buffer.handle, 0, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
}

GpuFrameProfiler::Scope GpuFrameProfiler::createScope(
    vk::CommandBuffer cb, const std::string &name, uint32_t index)
{
    const auto queryIndex = asserted_cast<uint32_t>(_queryScopeIndices.size());
    _queryScopeIndices.push_back(index);
    return Scope{cb, _queryPool, name, queryIndex};
}

std::vector<GpuFrameProfiler::ScopeTime> GpuFrameProfiler::getTimes()
{
    const auto timestampPeriodNanos = static_cast<double>(
        _device->properties().device.limits.timestampPeriod);

    // This is garbage if no frame has been completed with this profiler yet
    // Caller should make sure that isn't an issue.
    auto *mapped = reinterpret_cast<uint64_t *>(_buffer.mapped);

    std::vector<ScopeTime> times;
    times.reserve(_queryScopeIndices.size());
    for (auto i = 0u; i < _queryScopeIndices.size(); ++i)
    {
        // All bits valid should have been asserted on device creation
        auto start = mapped[static_cast<size_t>(i) * 2];
        auto end = mapped[static_cast<size_t>(i) * 2 + 1];
        auto nanos = (end - start) * timestampPeriodNanos;

        times.push_back(ScopeTime{
            .index = _queryScopeIndices[i],
            .millis = static_cast<float>(nanos * 1e-6),
        });
    };

    return times;
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

CpuFrameProfiler::Scope &CpuFrameProfiler::Scope::operator=(
    CpuFrameProfiler::Scope &&other) noexcept
{
    if (this != &other)
    {
        _start = other._start;
        _output = other._output;

        other._output = nullptr;
    }
    return *this;
}

CpuFrameProfiler::CpuFrameProfiler() { _nanos.reserve(sMaxScopeCount); }

void CpuFrameProfiler::startFrame() { _nanos.clear(); }

[[nodiscard]] CpuFrameProfiler::Scope CpuFrameProfiler::createScope(
    uint32_t index)
{
    assert(
        index == _nanos.size() &&
        "CpuFrameProfiler expects that all indices have a scope attached");
    (void)index;

    _nanos.emplace_back();

    return Scope{&_nanos.back()};
}

std::vector<CpuFrameProfiler::ScopeTime> CpuFrameProfiler::getTimes()
{
    std::vector<ScopeTime> times;
    times.reserve(_nanos.size());
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

Profiler::Profiler(Device *device, uint32_t maxFrameCount)
{
    _previousScopeNames.resize(maxFrameCount);
    _previousCpuScopeTimes.resize(maxFrameCount);

    _currentFrameScopeNames.reserve(sMaxScopeCount);

    for (auto i = 0u; i < maxFrameCount; ++i)
    {
        _gpuFrameProfilers.emplace_back(device);
        _previousScopeNames[i].reserve(sMaxScopeCount);
        _previousCpuScopeTimes[i].reserve(sMaxScopeCount);
    }
}

void Profiler::startCpuFrame()
{
    assert(_debugState == DebugState::NewFrame);

    // Only clear transients for this profiling frame. We'll figure out which
    // frame's data we'll overwrite in endCpuFrame, when we know the gpu frame
    // index of this frame
    _currentFrameScopeNames.clear();

    _cpuFrameProfiler.startFrame();

#ifndef NDEBUG
    _debugState = DebugState::StartCpuCalled;
#endif // NDEBUG
}

void Profiler::startGpuFrame(uint32_t frameIndex)
{
    assert(_debugState == DebugState::StartCpuCalled);
    assert(frameIndex < _gpuFrameProfilers.size());

    _currentFrame = frameIndex;

    // Store times from the previous iteration of this gpu frame index. We need
    // to read these before startFrame as that will reset the queries.
    _previousGpuScopeTimes = _gpuFrameProfilers[_currentFrame].getTimes();

    _gpuFrameProfilers[_currentFrame].startFrame();

#ifndef NDEBUG
    _debugState = DebugState::StartGpuCalled;
#endif // NDEBUG
}

void Profiler::endGpuFrame(vk::CommandBuffer cb)
{
    assert(_debugState == DebugState::StartGpuCalled);

    _gpuFrameProfilers[_currentFrame].endFrame(cb);

#ifndef NDEBUG
    _debugState = DebugState::EndGpuCalled;
#endif // NDEBUG
}

void Profiler::endCpuFrame()
{
    assert(_debugState == DebugState::EndGpuCalled);
    assert(_currentFrame < _previousScopeNames.size());
    assert(_currentFrame < _previousCpuScopeTimes.size());

    // We now know which frame's data we gave out in getTimes() so let's
    // overwrite them
    _previousScopeNames[_currentFrame] = _currentFrameScopeNames;
    _previousCpuScopeTimes[_currentFrame] = _cpuFrameProfiler.getTimes();

#ifndef NDEBUG
    _debugState = DebugState::NewFrame;
#endif // NDEBUG
}

Profiler::Scope Profiler::createCpuGpuScope(
    vk::CommandBuffer cb, std::string const &name)
{
    assert(_debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(_currentFrameScopeNames.size());
    assert(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _currentFrameScopeNames.push_back(name);

    return Scope{
        _gpuFrameProfilers[_currentFrame].createScope(cb, name, index),
        _cpuFrameProfiler.createScope(index)};
}

Profiler::Scope Profiler::createCpuScope(std::string const &name)
{
    assert(
        _debugState == DebugState::StartCpuCalled ||
        _debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(_currentFrameScopeNames.size());
    assert(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _currentFrameScopeNames.push_back(name);

    return Scope{_cpuFrameProfiler.createScope(index)};
}

std::vector<Profiler::ScopeTime> Profiler::getPreviousTimes()
{
    assert(_debugState == DebugState::StartGpuCalled);

    const auto &scopeNames = _previousScopeNames[_currentFrame];
    // This also handles the first calls before any scopes are recorded for the
    // frame index and the gpu data is garbage. Rest of the calls should be with
    // valid data as we have waited for swap with the corresponding frame index
    if (scopeNames.empty())
        return {};

    std::vector<ScopeTime> times;
    times.reserve(scopeNames.size());
    for (const auto &n : scopeNames)
        times.push_back(ScopeTime{
            .name = n,
        });

    for (const auto &t : _previousGpuScopeTimes)
    {
        assert(t.index < times.size());
        times[t.index].gpuMillis = t.millis;
    }

    for (const auto &t : _previousCpuScopeTimes[_currentFrame])
    {
        assert(t.index < times.size());
        times[t.index].cpuMillis = t.millis;
    }

    return times;
}
