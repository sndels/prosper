set(SHADER_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/shader.frag
    ${CMAKE_CURRENT_LIST_DIR}/shader.vert
    ${CMAKE_CURRENT_LIST_DIR}/skybox.frag
    ${CMAKE_CURRENT_LIST_DIR}/skybox.vert
)

foreach(GLSL ${SHADER_SOURCES})
    get_filename_component(FILENAME ${GLSL} NAME)
    set(SPIRV ${PROJECT_BINARY_DIR}/shader/${FILENAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/shader/"
        COMMAND glslangValidator -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
    )
    list(APPEND SPIRV_BINARIES ${SPIRV})
endforeach(GLSL)

add_custom_target(spirv-shaders DEPENDS ${SPIRV_BINARIES})
