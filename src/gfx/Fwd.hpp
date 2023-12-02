#ifndef PROSPER_GFX_FWD_HPP
#define PROSPER_GFX_FWD_HPP

// DescriptorAllocator.hpp
class DescriptorAllocator;

// Device.hpp
class Device;
class FileIncluder; // TODO: Move inside Device.cpp
struct DeviceProperties;
struct MemoryAllocationBytes;
struct QueueFamilies;

// Resources.hpp
struct AccelerationStructure;
struct Buffer;
struct BufferCreateInfo;
struct BufferDescription;
struct Image;
struct ImageCreateInfo;
struct ImageDescription;
struct TexelBuffer;
struct TexelBufferCreateInfo;
struct TexelBufferDescription;

// RingBuffer.hpp
class RingBuffer;

// ShaderReflection.hpp
class ShaderReflection;
struct DescriptorSetMetadata;

// Swapchain.hpp
class Swapchain;
struct SwapchainConfig;
struct SwapchainImage;
struct SwapchainSupport;

#endif // PROSPER_GFX_FWD_HPP
