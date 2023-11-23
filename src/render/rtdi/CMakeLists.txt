set(PROSPER_RTDI_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/RtDiInitialReservoirs.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDirectIllumination.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDiSpatialReuse.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDiTrace.hpp
    PARENT_SCOPE
)

set(PROSPER_RTDI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/RtDiInitialReservoirs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDirectIllumination.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDiSpatialReuse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDiTrace.cpp
    PARENT_SCOPE
)
