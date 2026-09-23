cmake_minimum_required(VERSION 3.16)

project(MALG LANGUAGES CXX)

# Global C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Parthenon integration
# ---------------------------------------------------------------------------
# Assume Parthenon lives in external/parthenon
# Turn off Parthenon's own examples and tests to keep your build lean
set(PARTHENON_DISABLE_EXAMPLES ON  CACHE BOOL "" FORCE)
set(PARTHENON_ENABLE_TESTING   OFF CACHE BOOL "" FORCE)

add_subdirectory(external/parthenon)

# ---------------------------------------------------------------------------
# MALG library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# MALG tests and examples (independent flags)
# ---------------------------------------------------------------------------
option(MALG_ENABLE_TESTING "Enable MALG tests (CTest + Catch2)" ON)
option(MALG_BUILD_EXAMPLES "Build MALG example executables" ON)

# Tests
if(MALG_ENABLE_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(MALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# MALG formatting support (clang-format) over project sources
# ---------------------------------------------------------------------------
option(MALG_ENABLE_FORMATTING "Enable MALG formatting targets (clang-format)" ON)

if(MALG_ENABLE_FORMATTING)
    find_program(CLANG_FORMAT_EXE
        NAMES clang-format clang-format-18 clang-format-17 clang-format-16
    )

    if(NOT CLANG_FORMAT_EXE)
        message(WARNING "clang-format not found; MALG formatting targets will be unavailable")
    else()
        # Collect all C/C++ sources and headers under src/, examples/, tests/
        # (but not external/ or build directories).
        file(GLOB_RECURSE MALG_FORMAT_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"

            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.c"

            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.h"

            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"

            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
        )

        list(LENGTH MALG_FORMAT_SOURCES _fmt_count)
        if(_fmt_count EQUAL 0)
            message(WARNING "No MALG sources found for formatting")
        else()
            # In-place formatting
            add_custom_target(format
                COMMAND "${CLANG_FORMAT_EXE}" -i ${MALG_FORMAT_SOURCES}
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                COMMENT "Running clang-format on MALG sources"
            )

            # Check-only formatting (CI-friendly)
            add_custom_target(check-format
                COMMAND "${CLANG_FORMAT_EXE}" --dry-run --Werror ${MALG_FORMAT_SOURCES}
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                COMMENT "Checking clang-format on MALG sources"
            )
        endif()
    endif()
endif()
