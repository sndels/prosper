#ifndef PROSPER_RENDER_FWD_HPP
#define PROSPER_RENDER_FWD_HPP

namespace render
{

namespace bloom
{

// bloom/Bloom.hpp
class Bloom;

// Rest of the namespace types are internal

} // namespace bloom

namespace dof
{

// dof/DepthOfField.hpp
class DepthOfField;

// Rest of the namespace types are internal

} // namespace dof

namespace rtdi
{

// rtdi/RtDirectIllumination.hpp
class RtDirectIllumination;

// Rest of the namespace types are internal

} // namespace rtdi

// ComputePass.hpp
class ComputePass;

// DebugRenderer.hpp
class DebugRenderer;

// DrawStats.hpp
struct DrawStats;

// DeferredShading.hpp
class DeferredShading;

// ForwardRenderer.hpp
class ForwardRenderer;

// GBuffer.hpp
struct GBuffer;

// GBufferRenderer.hpp
class GBufferRenderer;

// HierarchicalDepthDownsampler.hpp
class HierarchicalDepthDownsampler;

// ImageBasedLighting.hpp
class ImageBasedLighting;

// ImGuiRenderer.hpp
class ImGuiRenderer;

// LightClustering.hpp
class LightClustering;

// LightClustering.hpp
class LightClustering;
struct LightClusteringOutput;

// MeshletCuller.hpp
struct MeshletCullerSecondPhaseInputBuffers;
struct MeshletCullerFirstPhaseOutput;
struct MeshletCullerSecondPhaseOutput;
class MeshletCuller;

// Renderer.hpp
class Renderer;

// RenderResources.hpp
class RenderResources;

// RtReference.hpp
class RtReference;

// SkyboxRenderer.hpp
class SkyboxRenderer;

// TemporalAntiAliasing.hpp
class TemporalAntiAliasing;

// TextureDebug.hpp
class TextureDebug;

// TextureReadback.hpp
class TextureReadback;

// ToneMap.hpp
class ToneMap;

} // namespace render

#endif // PROSPER_RENDER_FWD_HPP
