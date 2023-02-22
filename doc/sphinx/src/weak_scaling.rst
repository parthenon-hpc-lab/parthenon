Running a Weak Scaling Test in Parthenon
========================================

-  Here we present how to perform a weak scaling test on a Power9
   architecture with two Volta GPUs per node.

-  We use the advection test with slightly modified input parameters for
   performance. AMR is turned on with 3 levels.

-  The following procedure was tested on power9 nodes on Darwin by Jonah
   Miller. However, the same procedure should hold more generically.

To Build
--------

-  Note that depending on your system you may need to disable HDF5 with
   ``-DPARTHENON_DISABLE_HDF5=On``.

.. code:: bash

   module purge
   module load cmake gcc/7.4.0 cuda/10.2 openmpi/p9/4.0.1-gcc_7.4.0 anaconda/Anaconda3.2019.10
   git clone git@github.com:parthenon-hpc-lab/parthenon.git --recursive
   cd parthenon
   git checkout develop && git pull
   git submodule update --init --recursive
   mkdir -p bin && cd bin
   cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=True -DKokkos_ARCH_POWER9=True -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper ..
   make -j

To Run
------

-  Note that if you disabled HDF5 in the previous step, you must open up
   the ``parthinput.advection`` file and comment out all blocks
   beginning with ``<parthenon/output*>``.

-  Place the following in your job script.

.. code:: bash

   N=1
   mpirun -np ${N} ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=64 parthenon/mesh/nx2=64 parthenon/mesh/
   nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 | tee ${N}.out

   N=2
   mpirun -np ${N} ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/
   nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 | tee ${N}.out

   N=4
   mpirun -np ${N} ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=128 parthenon/mesh/
   nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 | tee ${N}.out

   N=8
   mpirun -np ${N} ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=128 parthenon/mesh/
   nx3=128 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 | tee ${N}.out

   N=16
   mpirun -np ${N} ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=256 parthenon/mesh/nx2=128 parthenon/mesh/
   nx3=128 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 | tee ${N}.out

   # and so on

To get the timing data
----------------------

-  We use the built in instrumentation inside Parthenon, which is stored
   in the output.

.. code:: bash

   filename=timings.dat
   printf "# nprocs\tzone-cycles/cpu-second\n" > ${filename}
   # make sure upper bound on this is log2(Nprocs max)
   for n in {0..4}; do echo $((2**n)) $(grep "zone-cycles/cpu_second = " $((2**n)).out | cut -d "=" -f 2) >> ${filename}; done

You can now load the timings in your favorite plotting program.
