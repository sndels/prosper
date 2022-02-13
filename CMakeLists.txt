cmake_minimum_required(VERSION 3.20)
project(prosper)

if (MSVC)
add_compile_options(/MP)
endif() # MSVC

# Set up sub-builds and sources
add_subdirectory(ext)
add_subdirectory(include)
add_subdirectory(src)


# Set up project targets
add_executable(prosper ${PROSPER_SOURCES} ${PROSPER_INCLUDES})
target_compile_features(prosper
    PRIVATE
    cxx_std_20
)
if (MSVC)
    # From cppbestpractices
    target_compile_options(prosper
        PRIVATE
        /permissive-
        /W4
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14546
        /w14547
        /w14549
        /w14555
        /w14619
        /w14640
        /w14826
        /w14905
        /w14906
        /w14928
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
)
target_link_libraries(prosper
    PRIVATE
    glfw
    gli
    glm
    imgui
    shaderc
    stb_image
    vma
    vulkan
)
target_compile_definitions(prosper
    PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    # Set up absolute path to resources and binaries
    RES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/"
    BIN_PATH="${CMAKE_CURRENT_BINARY_DIR}/")
