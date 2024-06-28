#include "Profiler.hpp"

#include "../gfx/Device.hpp"
#include "Utils.hpp"

using namespace wheels;

namespace
{

// Each scope has a query for start and end
constexpr uint32_t sMaxTimestampCount = sMaxScopeCount * 2;
// TODO: Mesh shader stats
constexpr vk::QueryPipelineStatisticFlags sPipelineStatisticsFlags =
    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
    vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
// Use VkFlags directly, should be a typedef to an integral type
constexpr size_t sStatTypeCount = asserted_cast<size_t>(
    std::popcount(static_cast<VkFlags>(sPipelineStatisticsFlags)));

} // namespace

// This used everywhere and init()/destroy() order relative to other similar
// globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Profiler gProfiler;

GpuFrameProfiler::Scope::Scope(
    vk::CommandBuffer cb, QueryPools pools, const char *name,
    uint32_t queryIndex, bool includeStatistics)
: m_cb{cb}
, m_pools{pools}
, m_queryIndex{queryIndex}
, m_hasStatistics{includeStatistics}
{
    cb.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = name,
    });
    cb.writeTimestamp2(
        vk::PipelineStageFlagBits2::eTopOfPipe, m_pools.timestamps,
        m_queryIndex * 2);
    if (m_hasStatistics)
        cb.beginQuery(
            m_pools.statistics, m_queryIndex, vk::QueryControlFlags{});
}

GpuFrameProfiler::Scope::~Scope()
{
    if (m_cb)
    {
        if (m_hasStatistics)
            m_cb.endQuery(m_pools.statistics, m_queryIndex);
        m_cb.writeTimestamp2(
            vk::PipelineStageFlagBits2::eBottomOfPipe, m_pools.timestamps,
            m_queryIndex * 2 + 1);
        m_cb.endDebugUtilsLabelEXT();
    }
}

GpuFrameProfiler::Scope::Scope(GpuFrameProfiler::Scope &&other) noexcept
: m_cb{other.m_cb}
, m_pools{other.m_pools}
, m_queryIndex{other.m_queryIndex}
, m_hasStatistics{other.m_hasStatistics}
{
    other.m_cb = vk::CommandBuffer{};
}

GpuFrameProfiler::~GpuFrameProfiler() { destroy(); }

GpuFrameProfiler::GpuFrameProfiler(GpuFrameProfiler &&other) noexcept
: m_initialized{other.m_initialized}
, m_timestampBuffer{WHEELS_MOV(other.m_timestampBuffer)}
, m_statisticsBuffer{WHEELS_MOV(other.m_statisticsBuffer)}
, m_pools{other.m_pools}
, m_queryScopeIndices{WHEELS_MOV(other.m_queryScopeIndices)}
, m_scopeHasStats{WHEELS_MOV(other.m_scopeHasStats)}
{
    // Avoid dtor destroying what we moved
    other.m_timestampBuffer.handle = vk::Buffer{};
    other.m_statisticsBuffer.handle = vk::Buffer{};
    other.m_pools.statistics = vk::QueryPool{};
    other.m_pools.timestamps = vk::QueryPool{};
}

GpuFrameProfiler &GpuFrameProfiler::operator=(GpuFrameProfiler &&other) noexcept
{
    if (this != &other)
    {
        destroy();

        m_initialized = other.m_initialized;
        m_timestampBuffer = WHEELS_MOV(other.m_timestampBuffer);
        m_statisticsBuffer = WHEELS_MOV(other.m_statisticsBuffer);
        m_pools = other.m_pools;
        m_queryScopeIndices = WHEELS_MOV(other.m_queryScopeIndices);
        m_scopeHasStats = WHEELS_MOV(other.m_scopeHasStats);

        // Avoid dtor destroying what we moved
        other.m_timestampBuffer.handle = vk::Buffer{};
        other.m_statisticsBuffer.handle = vk::Buffer{};
        other.m_pools.statistics = vk::QueryPool{};
        other.m_pools.timestamps = vk::QueryPool{};
    }
    return *this;
}

void GpuFrameProfiler::init()
{
    WHEELS_ASSERT(!m_initialized);

    m_timestampBuffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeof(uint64_t) * sMaxTimestampCount,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "GpuProfilerTimestampReadback"});
    m_statisticsBuffer = gDevice.createBuffer(BufferCreateInfo{
        .desc =
            BufferDescription{
                .byteSize = sizeof(uint32_t) * sStatTypeCount * sMaxScopeCount,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent,
            },
        .debugName = "GpuProfilerStatisticsReadback"});
    m_pools.timestamps =
        gDevice.logical().createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = sMaxTimestampCount});
    m_pools.statistics =
        gDevice.logical().createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::ePipelineStatistics,
            .queryCount = sMaxScopeCount,
            .pipelineStatistics = sPipelineStatisticsFlags,
        });

    m_initialized = true;
}

void GpuFrameProfiler::destroy()
{
    // Don't check for m_initialized as we might be cleaning up after a failed
    // init.
    gDevice.logical().destroyQueryPool(m_pools.statistics);
    gDevice.logical().destroyQueryPool(m_pools.timestamps);
    gDevice.destroy(m_statisticsBuffer);
    gDevice.destroy(m_timestampBuffer);
}

void GpuFrameProfiler::startFrame()
{
    // Might be more optimal to do this in a command buffer if we had some other
    // use that was ensured to happen before all other command buffers.
    gDevice.logical().resetQueryPool(m_pools.timestamps, 0, sMaxTimestampCount);
    gDevice.logical().resetQueryPool(m_pools.statistics, 0, sMaxScopeCount);
    m_queryScopeIndices.clear();
    m_scopeHasStats.clear();
}

void GpuFrameProfiler::endFrame(vk::CommandBuffer cb)
{
    cb.copyQueryPoolResults(
        m_pools.timestamps, 0,
        asserted_cast<uint32_t>(m_queryScopeIndices.size() * 2),
        m_timestampBuffer.handle, 0, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    cb.copyQueryPoolResults(
        m_pools.statistics, 0,
        asserted_cast<uint32_t>(m_queryScopeIndices.size()),
        m_statisticsBuffer.handle, 0, sizeof(uint32_t) * sStatTypeCount,
        vk::QueryResultFlags{});
}

GpuFrameProfiler::Scope GpuFrameProfiler::createScope(
    vk::CommandBuffer cb, const char *name, uint32_t index,
    bool includeStatistics)
{
    const auto queryIndex = asserted_cast<uint32_t>(m_queryScopeIndices.size());
    m_queryScopeIndices.push_back(index);
    m_scopeHasStats.push_back(includeStatistics);
    return Scope{cb, m_pools, name, queryIndex, includeStatistics};
}

Array<GpuFrameProfiler::ScopeData> GpuFrameProfiler::getData(Allocator &alloc)
{
    const auto timestampPeriodNanos =
        static_cast<double>(gDevice.properties().device.limits.timestampPeriod);

    // This is garbage if no frame has been completed with this profiler yet
    // Caller should make sure that isn't an issue.
    const auto *timestamps = static_cast<uint64_t *>(m_timestampBuffer.mapped);
    const auto *stats = static_cast<uint32_t *>(m_statisticsBuffer.mapped);

    const size_t scopeCount = m_queryScopeIndices.size();
    WHEELS_ASSERT(scopeCount == m_scopeHasStats.size());
    Array<ScopeData> ret{alloc, scopeCount};
    for (auto i = 0u; i < scopeCount; ++i)
    {
        // All bits valid should have been asserted on device creation
        const uint64_t start = timestamps[static_cast<size_t>(i) * 2];
        const uint64_t end = timestamps[static_cast<size_t>(i) * 2 + 1];
        const double nanos =
            static_cast<double>(end - start) * timestampPeriodNanos;
        const float millis = static_cast<float>(nanos * 1e-6);
        const bool hasStats = m_scopeHasStats[i];

        Optional<PipelineStatistics> scopeStats;
        if (hasStats)
            scopeStats = PipelineStatistics{
                .clipPrimitives =
                    stats[static_cast<size_t>(i) * sStatTypeCount],
                .fragInvocations =
                    stats[static_cast<size_t>(i) * sStatTypeCount + 1],
            };

        ret.push_back(ScopeData{
            .index = m_queryScopeIndices[i],
            .millis = millis,
            .stats = scopeStats,
        });
    };

    return ret;
}

CpuFrameProfiler::Scope::Scope(std::chrono::nanoseconds *output)
: m_start{std::chrono::high_resolution_clock::now()}
, m_output{output}
{
}

CpuFrameProfiler::Scope::~Scope()
{
    if (m_output != nullptr)
        *m_output = std::chrono::high_resolution_clock::now() - m_start;
}

CpuFrameProfiler::Scope::Scope(CpuFrameProfiler::Scope &&other) noexcept
: m_start{other.m_start}
, m_output{other.m_output}
{
    other.m_output = nullptr;
}

CpuFrameProfiler::~CpuFrameProfiler()
{
    WHEELS_ASSERT(!m_initialized && "destroy() not called");
}

void CpuFrameProfiler::init()
{
    WHEELS_ASSERT(!m_initialized);

    m_queryScopeIndices.reserve(sMaxScopeCount);
    m_nanos.reserve(sMaxScopeCount);

    m_initialized = true;
}

void CpuFrameProfiler::destroy()
{
    // Clean up manually as we need to free things before allocator destroy()s
    // are called
    m_queryScopeIndices.~Array();
    m_nanos.~Array();

    m_initialized = false;
}

void CpuFrameProfiler::startFrame()
{
    m_queryScopeIndices.clear();
    m_nanos.clear();
}

[[nodiscard]] CpuFrameProfiler::Scope CpuFrameProfiler::createScope(
    uint32_t index)
{
    m_queryScopeIndices.push_back(index);
    m_nanos.emplace_back();

    return Scope{&m_nanos.back()};
}

Array<CpuFrameProfiler::ScopeTime> CpuFrameProfiler::getTimes(Allocator &alloc)
{
    size_t const scopeCount = m_queryScopeIndices.size();
    WHEELS_ASSERT(scopeCount == m_nanos.size());
    Array<ScopeTime> times{alloc, sMaxScopeCount};
    for (size_t i = 0; i < scopeCount; ++i)
    {
        const ScopeTime st{
            .index = m_queryScopeIndices[i],
            .millis =
                std::chrono::duration<float, std::milli>(m_nanos[i]).count(),
        };
        times.push_back(st);
    };

    return times;
}

Profiler::~Profiler()
{
    // This is a global with tricky destruction order relative to others so
    // require manual destroy();
    WHEELS_ASSERT(!m_initialized && "destroy() not called");
}

void Profiler::init()
{
    // This is a global and allocators are initialized in main() so we can't
    // reserve these in default member initializers.
    m_cpuFrameProfiler.init();
    m_gpuFrameProfilers.reserve(MAX_FRAMES_IN_FLIGHT);
    m_currentFrameScopeNames.reserve(sMaxScopeCount);
    m_previousScopeNames.reserve(MAX_FRAMES_IN_FLIGHT);
    m_previousCpuScopeTimes.reserve(MAX_FRAMES_IN_FLIGHT);
    m_previousGpuScopeData.reserve(sMaxScopeCount);

    for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        m_gpuFrameProfilers.emplace_back();
        m_gpuFrameProfilers.back().init();

        m_previousScopeNames.emplace_back(gAllocators.general, sMaxScopeCount);
        m_previousCpuScopeTimes.emplace_back(
            gAllocators.general, sMaxScopeCount);
    }

    m_initialized = true;
}

void Profiler::destroy()
{
    // Clean up manually as we need to free things before allocator destroy()s
    // are called
    m_cpuFrameProfiler.destroy();
    m_gpuFrameProfilers.~Array();
    m_currentFrameScopeNames.~Array();
    m_previousScopeNames.~Array();
    m_previousCpuScopeTimes.~Array();
    m_previousGpuScopeData.~Array();
    m_initialized = false;
}

void Profiler::startCpuFrame()
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::NewFrame);

    // Only clear transients for this profiling frame. We'll figure out which
    // frame's data we'll overwrite in endCpuFrame, when we know the gpu frame
    // index of this frame
    m_currentFrameScopeNames.clear();

    m_cpuFrameProfiler.startFrame();

    m_debugState = DebugState::StartCpuCalled;
}

void Profiler::startGpuFrame(uint32_t frameIndex)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::StartCpuCalled);
    WHEELS_ASSERT(frameIndex < m_gpuFrameProfilers.size());

    m_currentFrame = frameIndex;

    // Store data from the previous iteration of this gpu frame index. We need
    // to read these before startFrame as that will reset the queries.
    m_previousGpuScopeData =
        m_gpuFrameProfilers[m_currentFrame].getData(gAllocators.general);

    m_gpuFrameProfilers[m_currentFrame].startFrame();

    m_debugState = DebugState::StartGpuCalled;
}

void Profiler::endGpuFrame(vk::CommandBuffer cb)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::StartGpuCalled);

    m_gpuFrameProfilers[m_currentFrame].endFrame(cb);

    m_debugState = DebugState::EndGpuCalled;
}

void Profiler::endCpuFrame()
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::EndGpuCalled);
    WHEELS_ASSERT(m_currentFrame < m_previousScopeNames.size());
    WHEELS_ASSERT(m_currentFrame < m_previousCpuScopeTimes.size());

    // We now know which frame's data we gave out in getTimes() so let's
    // overwrite them
    m_previousScopeNames[m_currentFrame].clear();
    for (const auto &name : m_currentFrameScopeNames)
        m_previousScopeNames[m_currentFrame].emplace_back(
            gAllocators.general, name);
    m_previousCpuScopeTimes[m_currentFrame] =
        m_cpuFrameProfiler.getTimes(gAllocators.general);

    m_debugState = DebugState::NewFrame;
}

Profiler::Scope Profiler::createGpuScope(
    vk::CommandBuffer cb, const char *name, bool includeStatistics)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(m_currentFrameScopeNames.size());
    WHEELS_ASSERT(index < sMaxScopeCount && "Ran out of per-frame scopes");

    m_currentFrameScopeNames.emplace_back(gAllocators.general, name);

    return Scope{m_gpuFrameProfilers[m_currentFrame].createScope(
        cb, name, index, includeStatistics)};
}

Profiler::Scope Profiler::createCpuGpuScope(
    vk::CommandBuffer cb, const char *name, bool includeStatistics)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(m_currentFrameScopeNames.size());
    WHEELS_ASSERT(index < sMaxScopeCount && "Ran out of per-frame scopes");

    m_currentFrameScopeNames.emplace_back(gAllocators.general, name);

    return Scope{
        m_gpuFrameProfilers[m_currentFrame].createScope(
            cb, name, index, includeStatistics),
        m_cpuFrameProfiler.createScope(index)};
}

Profiler::Scope Profiler::createCpuScope(const char *name)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(
        m_debugState == DebugState::StartCpuCalled ||
        m_debugState == DebugState::StartGpuCalled);

    const auto index = asserted_cast<uint32_t>(m_currentFrameScopeNames.size());
    WHEELS_ASSERT(index < sMaxScopeCount && "Ran out of per-frame scopes");

    m_currentFrameScopeNames.emplace_back(gAllocators.general, name);

    return Scope{m_cpuFrameProfiler.createScope(index)};
}

Array<Profiler::ScopeData> Profiler::getPreviousData(Allocator &alloc)
{
    WHEELS_ASSERT(m_initialized);
    WHEELS_ASSERT(m_debugState == DebugState::StartGpuCalled);

    const auto &scopeNames = m_previousScopeNames[m_currentFrame];
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

    for (const auto &data : m_previousGpuScopeData)
    {
        WHEELS_ASSERT(data.index < ret.size());
        ret[data.index].gpuMillis = data.millis;
        ret[data.index].gpuStats = data.stats;
    }

    for (const auto &t : m_previousCpuScopeTimes[m_currentFrame])
    {
        WHEELS_ASSERT(t.index < ret.size());
        ret[t.index].cpuMillis = t.millis;
    }

    return ret;
}
