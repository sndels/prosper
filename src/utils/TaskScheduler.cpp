#include "TaskScheduler.hpp"

#include "Utils.hpp"

#include <fmt/format.h>
#include <wheels/assert.hpp>

namespace utils
{

// init()/destroy() order relative to other similar globals is handled in main()
// The constructor isn't tagged noexcept but it doesn't throw
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp)
enki::TaskScheduler gTaskScheduler;

void initTaskScheduler()
{
    gTaskScheduler.Initialize();

    WHEELS_ASSERT(
        utils::gTaskScheduler.GetNumTaskThreads() ==
        enki::GetNumHardwareThreads());
    const uint32_t threadCount = enki::GetNumHardwareThreads();

    enki::LambdaPinnedTask setThreadName;
    for (uint32_t i = 1; i < threadCount; ++i)
    {
        setThreadName.m_Function = [i]()
        { setCurrentThreadName(fmt::format("prosper bg {}", i).c_str()); };
        setThreadName.threadNum = i;
        utils::gTaskScheduler.AddPinnedTask(&setThreadName);
        utils::gTaskScheduler.WaitforTask(&setThreadName);
    }
}

void destroyTaskScheduler()
{
    gTaskScheduler.WaitforAllAndShutdown();
    gTaskScheduler.~TaskScheduler();
}

} // namespace utils
