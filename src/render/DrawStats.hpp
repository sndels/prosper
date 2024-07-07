#ifndef PROSPER_RENDER_DRAW_STATS_HPP
#define PROSPER_RENDER_DRAW_STATS_HPP

#include <cstdint>

struct DrawStats
{
    uint32_t drawnMeshletCount{0};
    uint32_t rasterizedTriangleCount{0};
    uint32_t totalTriangleCount{0};
    uint32_t totalMeshletCount{0};
    uint32_t totalMeshCount{0};
    uint32_t totalModelCount{0};
};

#endif // PROSPER_RENDER_DRAW_STATS_HPP
