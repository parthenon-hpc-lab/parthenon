# Building Parthenon on Various Systems

## Ubuntu 20.04 LTS

The following procedure has been tested for an Ubuntu 20.04 LTS system:

```bash
# install dependencies
# openmpi is installed implicitly by the hdf5 install
sudo apt-get update
install cmake build-essentials libhdf5-openmpi-dev

# Clone parthenon, with submodules
git clone --recursive https://github.com/lanl/parthenon.git
# make a bin directory
mkdir bin
cd bin
# configure and build
cmake ..
cmake -j --build .
# run unit and regression tests
ctest -LE performance
# run performance tests
ctest -L performance
```

### List of cmake options:

   |           Option             | Default  | Type   | Description |
   | ---------------------------: | :------- | :----- | :---------- |
   |            PARTHENON\_NGHOST | 2        | String | Number of ghost cells |
   | PARTHENON\_SINGLE\_PRECISION | OFF      | Option | Enable single precision mode if requested |
   |     PARTHENON\_DISABLE\_HDF5 | OFF      | Option | HDF5 is enabled by default if found, set this to True to disable HDF5 |
   |      PARTHENON\_DISABLE\_MPI | OFF      | Option | MPI is enabled by default if found, set this to True to disable MPI |
   |   PARTHENON\_DISABLE\_OPENMP | OFF      | Option | OpenMP is enabled by default if found, set this to True to disable OpenMP |
   |   ENABLE\_COMPILER\_WARNINGS | OFF      | Option | Enable compiler warnings |
   |        TEST\_ERROR\_CHECKING | OFF      | Option | Enables the error checking unit test. This test will FAIL |
   |    TEST\_INTEL\_OPTIMIZATION | OFF      | Option | Test intel optimization and vectorization |
   |    CHECK\_REGISTRY\_PRESSURE | OFF      | Option | Check the registry pressure for Kokkos CUDA kernels |
   |               BUILD\_TESTING | ON       | Option | Multi-testing enablement |
   | PARTHENON\_DISABLE\_EXAMPLES | OFF      | Option | Toggle building of examples, if regression tests are on, drivers needed by the tests will still be built |
   |   ENABLE\_INTEGRATION\_TESTS | ${BUILD\_TESTING} | Option | Enable integration tests |
   |    ENABLE\_REGRESSION\_TESTS | ${BUILD\_TESTING} | Option | Enable regression tests |
   |  REGRESSION\_GOLD\_STANDARD\_VER | #     | Int    | Version of current gold standard file used in regression tests. Default is set to latest version matching the source. |
   | REGRESSION\_GOLD\_STANDARD\_HASH | SHA512=... | String | Hash value of gold standard file to be downloaded. Used to ensure that the download is not corrupted. |
   | REGRESSION\_GOLD\_STANDARD\_SYNC | ON    | Option | Create `gold_standard` target to download gold standard files |
   |          ENABLE\_UNIT\_TESTS | ${BUILD\_TESTING} | Option | Enable unit tests |
   |               CODE\_COVERAGE | OFF      | Option | Builds with code coverage flags |
   |       CMAKE\_INSTALL\_PREFIX | machine specific | String | Optional path for shared library installation |

### NB: CMake options prefixed with *PARTHENON\_* modify behavior.

## Building a shared library

An alternative to building Parthenon alongside a custom app (as in the examples)
is to first build Parthenon separately as a shared library and then link to it
in the app.

To build Parthenon as a shared library, provide a *CMAKE\_INSTALL\_PREFIX* path
to the desired install location to the Parthenon cmake call, then build and install
(note that `--build` and `--install` required CMake 3.15 or greater)

```bash
cmake -DCMAKE_INSTALL_PREFIX="$your_install_dir" $parthenon_source_dir
cmake --build . -- -j
cmake --install .
```

This will also install Kokkos in the same directory; in general, one must be
able to call Kokkos functions directly when writing performant Parthenon apps.

The below example makefile can be used to compile the *calculate\_pi* example by
linking to a prior shared library installation of Parthenon. Note that library
flags must be appropriate for the Parthenon installation; it is not enough to
simply provide *-lparthenon*.

```bash
PARTHENON_INSTALL=/path/to/your/parthenon/install
CC=g++
CCFLAGS = -g -std=c++14 -L${PARTHENON_INSTALL}/lib \
 -I${PARTHENON_INSTALL}/include/ \
 -I${PARTHENON_INSTALL}/include/kokkos/
LIB_FLAGS = -Wl,-rpath,${PARTHENON_INSTALL}/lib -lparthenon -lmpi -lkokkoscore -lhdf5 -ldl -lkokkoscontainers -lz -lpthread -lgomp -lmpi_cxx
CC_COMPILE = $(CC) $(CCFLAGS) -c
CC_LOAD = $(CC) $(CCFLAGS)
.cpp.o:
  $(CC_COMPILE) $*.cpp
EXE = pi_example
all: $(EXE)
SRC = calculate_pi.cpp pi_driver.cpp
OBJ = calculate_pi.o pi_driver.o
INC = calculate_pi.hpp pi_driver.hpp
$(OBJ): $(INC) makefile
$(EXE): $(OBJ) $(INC) makefile
  $(CC_LOAD) $(OBJ) $(LIB_FLAGS) -o $(EXE)
clean:
  $(RM) $(OBJ) $(EXE)
```
