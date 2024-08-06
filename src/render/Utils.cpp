#include "Utils.hpp"

#include "render/RenderResources.hpp"
#include "utils/Utils.hpp"

vk::Extent2D getExtent2D(ImageHandle image)
{
    const vk::Extent3D targetExtent =
        gRenderResources.images->resource(image).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    const vk::Extent2D ret{
        .width = targetExtent.width,
        .height = targetExtent.height,
    };

    return ret;
}

vk::Extent2D getRoundedUpHalfExtent2D(ImageHandle image)
{
    const vk::Extent3D targetExtent =
        gRenderResources.images->resource(image).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    const vk::Extent2D ret{
        .width = roundedUpQuotient(targetExtent.width, 2u),
        .height = roundedUpQuotient(targetExtent.height, 2u),
    };

    return ret;
}

vk::Rect2D getRect2D(ImageHandle image)
{
    const vk::Extent3D targetExtent =
        gRenderResources.images->resource(image).extent;
    WHEELS_ASSERT(targetExtent.depth == 1);

    const vk::Rect2D ret{
        .offset = {0, 0},
        .extent =
            {
                targetExtent.width,
                targetExtent.height,
            },
    };

    return ret;
}
