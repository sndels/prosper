cmake_minimum_required(VERSION 3.4)
project(prosper)

# Set up external dependencies
find_package(Vulkan)
message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

# Set up sub-builds and sources
add_subdirectory(ext)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(res/shader)

# Set up absolute path to resources
add_definitions(-DRES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")

# Set up project targets
add_executable(prosper ${PROSPER_SOURCES} ${PROSPER_INCLUDES})
target_compile_features(prosper
    PRIVATE
    cxx_std_17
)
if (MSVC)
target_compile_options(prosper
    PRIVATE
    -W4     # Wall is _very_ verbose
    -wd4201 # glm uses nameless structs/unions
    -wd4996 # yup, I do want to use fopen
)
else()
# Anon structs cannot be suppressed in g++ pedantic
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(prosper
        PRIVATE
        -Wall
        -Wunused-parameter
        -Wno-gnu-anonymous-struct # glm
        -Wno-nested-anon-types # glm
        -Wno-missing-braces
        -Werror
        -pedantic
    )
endif()
endif() # MSVC
target_include_directories(prosper
    PRIVATE
    ${PROSPER_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIR}
)
target_link_libraries(prosper
    PRIVATE
    glfw
    glm
    stb_image
    ${Vulkan_LIBRARY}
)
add_dependencies(prosper spirv-shaders)
