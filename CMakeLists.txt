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


# Set up project targets
add_executable(prosper ${PROSPER_SOURCES} ${PROSPER_INCLUDES})
target_compile_features(prosper
    PRIVATE
    cxx_std_20
)
if (MSVC)
    target_compile_options(prosper
        PRIVATE
        /wd4005 # glfw seems to misbehave
    )
else()
    target_compile_options(prosper
        PRIVATE
        -pedantic
        -Wall
        -Wextra
        -Wunused
    )
endif() # NOT MSVC
target_include_directories(prosper
    PRIVATE
    ${PROSPER_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIR}
)
target_link_libraries(prosper
    PRIVATE
    glfw
    gli
    glm
    imgui
    stb_image
    vma
    ${Vulkan_LIBRARY}
)
target_compile_definitions(prosper
    PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    # Set up absolute path to resources and binaries
    RES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/"
    BIN_PATH="${CMAKE_CURRENT_BINARY_DIR}/")

add_dependencies(prosper spirv-shaders)
