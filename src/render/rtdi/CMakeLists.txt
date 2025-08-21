set(PROSPER_RTDI_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/InitialReservoirs.hpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDirectIllumination.hpp
    ${CMAKE_CURRENT_LIST_DIR}/SpatialReuse.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Trace.hpp
    PARENT_SCOPE
)

set(PROSPER_RTDI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/InitialReservoirs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RtDirectIllumination.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SpatialReuse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Trace.cpp
    PARENT_SCOPE
)
