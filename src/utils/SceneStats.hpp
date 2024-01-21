
#ifndef PROSPER_UTILS_SCENE_STATS_HPP
#define PROSPER_UTILS_SCENE_STATS_HPP

#include <cstdint>

struct SceneStats
{
    uint32_t totalTriangleCount{0};
    uint32_t totalMeshletCount{0};
    uint32_t totalMeshCount{0};
    uint32_t totalNodeCount{0};
    uint32_t animatedNodeCount{0};
};

#endif // PROSPER_UTILS_SCENE_STATS_HPP
