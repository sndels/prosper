# prosper
![screenshot](screenshot.png)

Vulkan renderer spun off from following https://vulkan-tutorial.com/. Work of [Sascha Willems](https://github.com/SaschaWillems) and [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii) also used as reference.

Current features include:
* glTF 2.0 assets
    * Only a very limited subset is currently supported
* Physically based shading
* Tangent space normal mapping
* Mipmapping
* Transparency (no sorting)
* Skybox
* Uncharted 2 -tonemapping
* Clustered lighting
  * Points, spots
  * Sphere bounds
  * View space clusters with depth slices
* Bindless materials
* Vulkan RT
  * ID debug only for now

Depends externally on [Vulkan SDK](https://vulkan.lunarg.com/) and the dependencies of [glfw](https://github.com/glfw/glfw). Includes [glfw](https://github.com/glfw/glfw), [gli](https://github.com/g-truc/gli), [glm](https://github.com/g-truc/glm), [imgui](https://github.com/ocornut/imgui), [libshaderc](https://github.com/google/shaderc), [tinygltf](https://github.com/syoyo/tinygltf) and [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) as submodules.
