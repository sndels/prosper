#ifndef PROSPER_UTILS_TASK_SCHEDULER_HPP
#define PROSPER_UTILS_TASK_SCHEDULER_HPP

#include <TaskScheduler.h>

namespace utils
{
// init()/destroy() order relative to other similar globals is handled in main()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern enki::TaskScheduler gTaskScheduler;

void initTaskScheduler();
void destroyTaskScheduler();

} // namespace utils

#endif // PROSPER_UTILS_TASK_SCHEDULER_HPP
