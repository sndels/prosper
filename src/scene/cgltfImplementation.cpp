// MSVC /external -stuff doesn't apply here so drop warnings manually
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif // _MSC_VER

// NOLINTBEGIN

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

// NOLINTEND

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER
