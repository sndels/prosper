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

GpuFrameProfiler::Scope::Scope(GpuFrameProfiler::Scope &&other)
: _cb{other._cb}
, _queryPool{other._queryPool}
, _queryIndex{other._queryIndex}
{
    other._cb = vk::CommandBuffer{};
}

GpuFrameProfiler::Scope &GpuFrameProfiler::Scope::operator=(
    GpuFrameProfiler::Scope &&other)
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
    if (_device)
    {
        _device->logical().destroyQueryPool(_queryPool);
        _device->destroy(_buffer);
    }
}

GpuFrameProfiler::GpuFrameProfiler(GpuFrameProfiler &&other)
: _device{other._device}
, _buffer{other._buffer}
, _queryPool{other._queryPool}
, _queryScopeIndices{std::move(other._queryScopeIndices)}
{
    other._device = nullptr;
}

GpuFrameProfiler &GpuFrameProfiler::operator=(GpuFrameProfiler &&other)
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
        auto start = mapped[i * 2];
        auto end = mapped[i * 2 + 1];
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
    if (_output)
        *_output = std::chrono::high_resolution_clock::now() - _start;
}

CpuFrameProfiler::Scope::Scope(CpuFrameProfiler::Scope &&other)
: _start{other._start}
, _output{other._output}
{
    other._output = nullptr;
}

CpuFrameProfiler::Scope &CpuFrameProfiler::Scope::operator=(
    CpuFrameProfiler::Scope &&other)
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

CpuFrameProfiler::CpuFrameProfiler(CpuFrameProfiler &&other)
: _nanos{std::move(other._nanos)}
{
}

CpuFrameProfiler &CpuFrameProfiler::operator=(CpuFrameProfiler &&other)
{
    if (this != &other)
        _nanos = std::move(other._nanos);
    return *this;
}

void CpuFrameProfiler::startFrame() { _nanos.clear(); }

[[nodiscard]] CpuFrameProfiler::Scope CpuFrameProfiler::createScope(
    uint32_t index)
{
    assert(
        index == _nanos.size() &&
        "CpuFrameProfiler expects that all indices have a scope attached");
    (void)index;

    _nanos.push_back({});

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
    for (auto i = 0u; i < maxFrameCount; ++i)
    {
        _gpuFrameProfilers.emplace_back(device);
        _cpuFrameProfilers.emplace_back();
    }

    _frameScopeNames.resize(maxFrameCount);
}

std::vector<Profiler::ScopeTime> Profiler::startFrame(uint32_t index)
{
    assert(index < _gpuFrameProfilers.size());
    assert(index < _cpuFrameProfilers.size());
    assert(index < _frameScopeNames.size());

    const auto times = getTimes(index);

    _currentFrame = index;
    _frameScopeNames[_currentFrame].clear();

    _gpuFrameProfilers[_currentFrame].startFrame();
    _cpuFrameProfilers[_currentFrame].startFrame();

    return times;
}

void Profiler::endFrame(vk::CommandBuffer cb)
{
    _gpuFrameProfilers[_currentFrame].endFrame(cb);
}

Profiler::Scope Profiler::createCpuGpuScope(
    vk::CommandBuffer cb, std::string const &name)
{
    const auto index =
        asserted_cast<uint32_t>(_frameScopeNames[_currentFrame].size());
    assert(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _frameScopeNames[_currentFrame].push_back(name);

    return Scope{
        std::move(
            _gpuFrameProfilers[_currentFrame].createScope(cb, name, index)),
        std::move(_cpuFrameProfilers[_currentFrame].createScope(index))};
}

Profiler::Scope Profiler::createCpuScope(std::string const &name)
{
    const auto index =
        asserted_cast<uint32_t>(_frameScopeNames[_currentFrame].size());
    assert(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _frameScopeNames[_currentFrame].push_back(name);

    return Scope{
        std::move(_cpuFrameProfilers[_currentFrame].createScope(index))};
}

std::vector<Profiler::ScopeTime> Profiler::getTimes(uint32_t frameIndex)
{
    // This also handles the first calls before any scopes are recorded when
    // gpu data is garbage. Rest of the calls should be with valid data as we
    // have waited for swap with the corresponding frame index
    const auto &scopeNames = _frameScopeNames[frameIndex];
    if (scopeNames.size() == 0)
        return {};

    std::vector<ScopeTime> times;
    times.reserve(scopeNames.size());
    for (auto i = 0u; i < scopeNames.size(); ++i)
        times.push_back(ScopeTime{
            .name = scopeNames[i],
        });

    for (const auto &t : _gpuFrameProfilers[frameIndex].getTimes())
        times[t.index].gpuMillis = t.millis;

    for (const auto &t : _cpuFrameProfilers[frameIndex].getTimes())
        times[t.index].cpuMillis = t.millis;

    return times;
}
