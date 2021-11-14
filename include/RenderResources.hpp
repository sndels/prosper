#ifndef PROSPER_RENDER_RESOURCES_HPP
#define PROSPER_RENDER_RESOURCES_HPP

#include "Device.hpp"

// Renderpasses that create the resources are responsible for their recreation,
// lifetime
struct RenderResources
{
    Image sceneColor;
    Image sceneDepth;
};

#endif // PROSPER_RENDER_RESOURCES_HPP
