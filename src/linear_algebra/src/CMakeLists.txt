# src/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(MALG_SOURCES
    "${SRC_DIR}/matrix.cpp"
)

set(MALG_HEADERS
    "${SRC_DIR}/implicit_qr.hpp"
    "${SRC_DIR}/givens.hpp"
    "${SRC_DIR}/householder.hpp"
    "${SRC_DIR}/matrix.hpp"
    "${SRC_DIR}/qr_decomposition.hpp"
    "${SRC_DIR}/square_svd.hpp"
    "${SRC_DIR}/symmetric_evd.hpp"
)

# MALG now a real static library that depends on Parthenon
add_library(MALG STATIC
    ${MALG_SOURCES}
    ${MALG_HEADERS}
)

target_include_directories(MALG
    PUBLIC
        ${SRC_DIR}
)

target_link_libraries(MALG
    PUBLIC
        Parthenon::parthenon
)

# Optional export to parent if you still want these variables
set(MALG_SOURCES  "${MALG_SOURCES}"  PARENT_SCOPE)
set(MALG_HEADERS  "${MALG_HEADERS}"  PARENT_SCOPE)