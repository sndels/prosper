// MSVC /external -stuff doesn't apply here so drop warnings manually
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

// NOLINTBEGIN

#define TINYGLTF_IMPLEMENTATION
#ifdef _WIN32
#define STBI_MSC_SECURE_CRT
#endif // _WIN32
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

// NOLINTEND

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER
