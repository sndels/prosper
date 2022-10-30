# Pull submodules
execute_process(COMMAND git submodule update --init --depth 1)

find_package(Vulkan)
message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

add_library(vulkan INTERFACE)
target_include_directories(vulkan SYSTEM INTERFACE ${Vulkan_INCLUDE_DIR})
target_link_libraries(vulkan INTERFACE ${Vulkan_LIBRARY})

# GLM is header only
add_library(glm INTERFACE)

# Define as system to suppress warnings
target_include_directories(glm SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/glm)

# Use ready CMakeLists when available
# Override glfw default options
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
add_subdirectory(glfw)
get_target_property(glfw_include_dirs glfw INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(glfw PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${glfw_include_dirs}")

message(STATUS "Sync shaderc dependencies")
execute_process(COMMAND
    python utils/git-sync-deps
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shaderc
)
set(SHADERC_SKIP_INSTALL ON CACHE BOOL "Skip installation")
set(SHADERC_SKIP_TESTS ON CACHE BOOL "Skip building tests")
set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "Skip building examples")
set(SHADERC_SKIP_COPYRIGHT_CHECK ON CACHE BOOL "Skip copyright check")
add_subdirectory(shaderc)
get_target_property(shaderc_include_dirs shaderc INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(shaderc PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${shaderc_include_dirs}")

add_library(imgui STATIC "")
target_compile_features(imgui
    PRIVATE
    cxx_std_11
)
target_sources(imgui
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui
    SYSTEM
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/imgui
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
)
target_link_libraries(imgui
    PUBLIC
    glfw
    vulkan
)

# tinygltf is header only
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# vulkan memory allocator is header only
add_library(vma INTERFACE)

# Define as system to suppress warnings
target_include_directories(vma SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/vma/include)

# gli is header only
add_library(gli INTERFACE)

# Define as system to suppress warnings
target_include_directories(gli SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/gli)

# CMP_Framework setup copied over from compressonator/cmp_framework
# as inlucding the whole  subdir doesn't work and the lib config is made assuming
# it is built as a part of Compressonator
# Also contains stb_image
add_library(CMP_Framework STATIC "")
target_compile_features(CMP_Framework
    PRIVATE
    cxx_std_17
)
target_compile_definitions(CMP_Framework PUBLIC
    _CMP_CPP17_=1
    ENABLE_MAKE_COMPATIBLE_API # Byte<->Float to make all source and dest compatible
    USE_OLD_SWIZZLE # Remove swizzle flag and abide by CMP_Formats
)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(CMP_Framework PUBLIC
        -DCMP_USE_XMMINTRIN WIN32_LEAN_AND_MEAN=1 _WIN32
    )
endif()

file(GLOB_RECURSE HALF_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/common/half/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/common/half/*.cpp
)
file(GLOB_RECURSE DDS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/cimage/dds/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/cimage/dds/*.cpp
)
file(GLOB_RECURSE HPC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_encode/hpc/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_encode/hpc/*.cpp
)
file(GLOB_RECURSE CORE_SHADERS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/shaders/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/shaders/*.cpp
)
file(GLOB_RECURSE CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/source/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/source/*.cpp
)
file(GLOB_RECURSE CMP_FRAMEWORK_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/*.cpp
)
file(GLOB_RECURSE CMP_FRAMEWORK_COMMON_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/common/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/common/*.cpp
)

target_sources(CMP_Framework
    PRIVATE
    ${HALF_SRC}
    ${DDS_SRC}
    ${HPC_SRC}
    ${CORE_SHADERS_SRC}
    ${CORE_SRC}
    ${CMP_FRAMEWORK_SRC}
    ${CMP_FRAMEWORK_COMMON_SRC}

    # CMP_Math
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/cmp_math/cmp_math_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/cmp_math/cmp_math_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/cmp_math/cmp_math_cpuid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/cmp_math/cmp_math_cpuid.cpp

    # CMP_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/atiformats.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/atiformats.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/pluginbase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/plugininterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/pluginmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/pluginmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/query_timer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/query_timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/stb_image.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/gltf/stb_image_write.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/tc_pluginapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/tc_plugininternal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/tc_plugininternal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/utilfuncs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/utilfuncs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/cmp_fileio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/cmp_fileio.cpp

    # CMP_SDK
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bcn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bcn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc1/bc1.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc1/bc1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc2/bc2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc2/bc2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc3/bc3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc3/bc3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc4/bc4.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc4/bc4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc5/bc5.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc5/bc5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc6/bc6h.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc6/bc6h.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc7/bc7.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk/bc7/bc7.cpp
)

target_include_directories(CMP_Framework
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_compressonatorlib
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_framework/common/half
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/source
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/cmp_core/shaders
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/ccmp_sdk
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_plugins/common/gltf
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/cmp_math
    ${CMAKE_CURRENT_SOURCE_DIR}/compressonator/applications/_libs/gpu_decode
)

if(UNIX)
    target_compile_definitions(CMP_Framework PRIVATE _LINUX)
endif()
