
#ifndef PROSPER_UTILS_SCENE_STATS_HPP
#define PROSPER_UTILS_SCENE_STATS_HPP

#include <cstdint>

namespace utils
{

struct SceneStats
{
    uint32_t totalNodeCount{0};
    uint32_t animatedNodeCount{0};
};

} // namespace utils

#endif // PROSPER_UTILS_SCENE_STATS_HPP
