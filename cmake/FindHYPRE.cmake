#[=======================================================================[.rst:
FindHYPRE
-------

Finds the HYPRE library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``HYPRE::HYPRE``
  The HYPRE library

We will try looking in the HYPRE_DIR user provided path in site.cmake

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``HYPRE_FOUND``
  True if the system has the HYPRE library.
``HYPRE_VERSION``
  The version of the HYPRE library which was found.
``HYPRE_INCLUDE_DIRS``
  Include directories needed to use HYPRE.
``HYPRE_LIBRARIES``
  Libraries needed to link to HYPRE.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``HYPRE_INCLUDE_DIR``
  The directory containing ``foo.h``.
``HYPRE_LIBRARY``
  The path to the HYPRE library.

#]=======================================================================]


find_path(HYPRE_INCLUDE_DIR NAMES HYPRE.h HINTS ${HYPRE_DIR}/include)
find_library(HYPRE_LIBRARY NAMES HYPRE HINTS ${HYPRE_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HYPRE
  FOUND_VAR HYPRE_FOUND
  REQUIRED_VARS
    HYPRE_LIBRARY
    HYPRE_INCLUDE_DIR
)

  # VERSION_VAR HYPRE_VERSION
if(HYPRE_FOUND AND NOT TARGET HYPRE::HYPRE)
  add_library(HYPRE::HYPRE UNKNOWN IMPORTED)
  set_target_properties(HYPRE::HYPRE PROPERTIES
    IMPORTED_LOCATION "${HYPRE_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${PC_HYPRE_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  HYPRE_INCLUDE_DIR
  HYPRE_LIBRARY
)


