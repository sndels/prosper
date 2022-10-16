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
    uint32_t index)
: _cb{cb}
, _queryPool{queryPool}
, _index{index}
{
    cb.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = name.c_str(),
    });
    cb.writeTimestamp2(
        vk::PipelineStageFlagBits2::eTopOfPipe, _queryPool, index * 2);
}

GpuFrameProfiler::Scope::~Scope()
{
    if (_cb)
    {
        _cb.writeTimestamp2(
            vk::PipelineStageFlagBits2::eBottomOfPipe, _queryPool,
            _index * 2 + 1);
        _cb.endDebugUtilsLabelEXT();
    }
}

GpuFrameProfiler::Scope::Scope(GpuFrameProfiler::Scope &&other)
: _cb{other._cb}
, _queryPool{other._queryPool}
, _index{other._index}
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
        _index = other._index;

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
, _writtenQueries{other._writtenQueries}
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
        _writtenQueries = other._writtenQueries;

        other._device = nullptr;
    }
    return *this;
}

void GpuFrameProfiler::startFrame()
{
    _writtenQueries = 0;
    // Might be more optimal to do this in a command buffer if we had some other
    // use that was ensured to happen before all other command buffers.
    _device->logical().resetQueryPool(_queryPool, 0, sMaxQueryCount);
}

void GpuFrameProfiler::endFrame(vk::CommandBuffer cb)
{
    cb.copyQueryPoolResults(
        _queryPool, 0, _writtenQueries, _buffer.handle, 0, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);
}

GpuFrameProfiler::Scope GpuFrameProfiler::createScope(
    vk::CommandBuffer cb, const std::string &name, uint32_t index)
{
    _writtenQueries += 2; // Assume both start and end are written
    return Scope{cb, _queryPool, name, index};
}

std::vector<GpuFrameProfiler::ScopeTime> GpuFrameProfiler::getTimes()
{
    const auto timestampPeriodNanos = static_cast<double>(
        _device->properties().device.limits.timestampPeriod);

    // This is garbage if no frame has been completed with this profiler yet
    // Caller should make sure that isn't an issue.
    auto *mapped = reinterpret_cast<uint64_t *>(_buffer.mapped);

    std::vector<ScopeTime> times;
    times.reserve(_writtenQueries / 2);
    for (auto i = 0u; i < _writtenQueries; i += 2)
    {
        // All bits valid should have been asserted on device creation
        auto start = mapped[i];
        auto end = mapped[i + 1];
        auto nanos = (end - start) * timestampPeriodNanos;

        times.push_back(ScopeTime{
            .index = i / 2,
            .millis = static_cast<float>(nanos * 1e-6),
        });
    };

    return times;
}

Profiler::Profiler(Device *device, uint32_t maxFrameCount)
{
    for (auto i = 0u; i < maxFrameCount; ++i)
        _gpuFrameProfilers.emplace_back(device);

    _scopeNames.resize(maxFrameCount);
}

std::vector<Profiler::ScopeTime> Profiler::startFrame(uint32_t index)
{
    assert(index < _gpuFrameProfilers.size());
    assert(index < _scopeNames.size());

    const auto times = getTimes(index);

    _currentFrame = index;
    _scopeNames[_currentFrame].clear();

    _gpuFrameProfilers[_currentFrame].startFrame();

    return times;
}

void Profiler::endFrame(vk::CommandBuffer cb)
{
    _gpuFrameProfilers[_currentFrame].endFrame(cb);
}

Profiler::Scope Profiler::createScope(
    vk::CommandBuffer cb, std::string const &name)
{
    const auto index =
        asserted_cast<uint32_t>(_scopeNames[_currentFrame].size());
    assert(index < sMaxScopeCount && "Ran out of per-frame scopes");

    _scopeNames[_currentFrame].push_back(name);

    return Scope{std::move(
        _gpuFrameProfilers[_currentFrame].createScope(cb, name, index))};
}

std::vector<Profiler::ScopeTime> Profiler::getTimes(uint32_t frameIndex)
{
    // This also handles the first calls before any scopes are recorded and
    // data is garbage. Rest of the calls should be with valid data as we have
    // waited for swap with the corresponding frame index
    const auto &scopeNames = _scopeNames[frameIndex];
    if (scopeNames.size() == 0)
        return {};

    const auto gpuTimes = _gpuFrameProfilers[frameIndex].getTimes();

    std::vector<ScopeTime> times;
    times.reserve(gpuTimes.size());
    for (const auto &t : gpuTimes)
    {
        times.push_back(ScopeTime{
            .name = scopeNames[t.index],
            .gpuMillis = t.millis,
        });
    }

    return times;
}
