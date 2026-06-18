Fourier Transforms
==================

Parthenon provides infrastructure for performing distributed Fast Fourier Transforms (FFTs)
on uniform meshes via the :cpp:class:`FFTManager` and :cpp:class:`UniformGridHelper` classes.
These are built on top of `heFFTe <https://github.com/icl-utk-edu/heffte>`_ and support
both CPU and GPU backends transparently.

.. note::
   FFT functionality requires ``num_packs = 1`` in the input file, meaning all meshblocks
   on a rank are packed into a single partition. This is required for the flat array indexing
   used by heFFTe.

Overview
--------

The FFT infrastructure consists of two classes that work together:

* :cpp:class:`FFTManager` — manages the FFT plan and performs forward/backward transforms
* :cpp:class:`UniformGridHelper` — provides mesh layout information and helper functions
  for mapping between Parthenon's meshblock-based data layout and the flat arrays required
  by heFFTe

Both are accessible via the :cpp:class:`Mesh` object:

.. code-block:: cpp

   auto fftManager        = pmesh->GetFFTManager();
   auto uniformGridHelper = pmesh->GetUniformGridHelper();

Both classes use the :cpp:struct:`Box3D` struct to describe spatial extents:

.. code-block:: cpp

   struct Box3D {
       int low[3];    // lower bound in each dimension
       int high[3];   // upper bound in each dimension
       int size[3];   // size in each dimension: high - low + 1
   };

Normalization Convention
------------------------

The forward transform applies a :math:`1/N^3` normalization, and the backward transform
applies no normalization. This means the round-trip (forward followed by backward) recovers
the original field exactly, and Parseval's theorem reads:

.. math::

   \sum_{\mathbf{k}} |\hat{f}(\mathbf{k})|^2 = \frac{1}{N^3} \sum_{\mathbf{x}} |f(\mathbf{x})|^2

Physical wavenumbers are related to integer mode numbers by :math:`k_\mathrm{phys} = 2\pi k / L`,
assuming a periodic domain of size :math:`L`.

Backends
--------

The backend is selected automatically at compile time based on the Kokkos execution space:

* **GPU** (CUDA/HIP): uses heFFTe's GPU backend
* **CPU**: uses heFFTe's CPU backend (FFTW or MKL if available, otherwise stock)

No code changes are required to switch between backends.

Basic Usage
-----------

The following example demonstrates the complete workflow for performing a forward and
backward FFT of a scalar field registered in Parthenon.

Allocating arrays
~~~~~~~~~~~~~~~~~

FFT input and output arrays are standard Parthenon device arrays. The sizes are provided
by :cpp:class:`FFTManager`:

.. code-block:: cpp

   const auto fft_size_inbox  = fftManager->size_real_space_box();
   const auto fft_size_outbox = fftManager->size_fourier_space_box();

   parthenon::ParArray1D<Real>                input("input",  fft_size_inbox);
   parthenon::ParArray1D<Kokkos::complex<Real>>  output("output", fft_size_outbox);
   parthenon::ParArray1D<Real>                result("result", fft_size_inbox);

Note that complex arrays must use Kokkos::complex, not std::complex, so that complex arithmetic is possible in Kokkos kernels.

Gathering a field from the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:func:`UniformGridHelper::GatherField` copies a single component of a named Parthenon
variable into a flat array suitable for FFT input:

.. code-block:: cpp

   // Gather component 0 of "cons" into the input array
   uniformGridHelper->GatherField("cons", 0, input);

For derived quantities that require computation,
use a custom gather loop with :cpp:func:`UniformGridHelper::GetKernelHelper`:

.. code-block:: cpp

   auto &md   = pmesh->mesh_data.Get();
   auto cons = md->PackVariables(std::vector<std::string>{"cons"});

   auto &mbb = uniformGridHelper->MeshBlockBox; // interior cell bounds within a meshblock

   auto helper = uniformGridHelper->GetKernelHelper();

   parthenon::par_for(
       "GatherVelocity", 0, md->NumBlocks() - 1,
       mbb.low[2], mbb.high[2],
       mbb.low[1], mbb.high[1],
       mbb.low[0], mbb.high[0],
       KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
           const auto idx = helper.FlatIndex(b, k, j, i);
           input(idx) = cons(b, 1, k, j, i) / cons(b, 0, k, j, i);
       });

Performing the transforms
~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:func:`FFTManager::Forward` and :cpp:func:`FFTManager::Backward` operate on raw
device pointers:

.. code-block:: cpp

   // Forward FFT (applies 1/N^3 normalization)
   fftManager->Forward(input.data(), output.data());

   // ... process output in Fourier space ...

   // Backward FFT (no normalization)
   fftManager->Backward(output.data(), result.data());

Processing in Fourier space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The local Fourier space box is accessible via :cpp:func:`FFTManager::fourier_space_box`.
Use :cpp:func:`FFTManager::GetKernelHelper` to obtain a device-copyable helper that
provides ``FourierFlatIndex`` and ``Wavevector``:

.. code-block:: cpp

   auto fft_helper = fftManager->GetKernelHelper();
   auto outbox     = fftManager->fourier_space_box();

   parthenon::par_for(
       "FourierSpaceKernel",
       outbox.low[2], outbox.high[2],
       outbox.low[1], outbox.high[1],
       outbox.low[0], outbox.high[0],
       KOKKOS_LAMBDA(const int kx3_idx, const int kx2_idx, const int kx1_idx) {

           const auto idx = fft_helper.FourierFlatIndex(kx3_idx, kx2_idx, kx1_idx);

           // integer wavevector components (negative frequencies unwrapped)
           auto [kx3, kx2, kx1] = fft_helper.Wavevector(kx3_idx, kx2_idx, kx1_idx);

           // ... process output[idx] ...
       });

.. note::
   The r2c transform only stores modes with :math:`k_{x1} \geq 0`. When computing
   quantities like the power spectrum, modes with :math:`0 < k_{x1} < n_{x1}/2` must be
   counted twice to account for Hermitian symmetry:

   .. code-block:: cpp

      const auto fac = ((kx1 > 0) && (2 * kx1 != nx1)) ? 2.0 : 1.0;

Scattering a field back to the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:func:`UniformGridHelper::ScatterField` copies a flat array back to a named
Parthenon variable on the mesh:

.. code-block:: cpp

   parthenon::ParArray1D<Real> result("result", fft_size_inbox);
   // ... fill result ...
   uniformGridHelper->ScatterField(result, "my_derived_field", 0);

The variable must be registered in the package before use (see :ref:`state`).

API Reference
-------------

Box3D
~~~~~

.. code-block:: cpp

   struct Box3D {
       int low[3];    // lower index bound in each dimension
       int high[3];   // upper index bound in each dimension
       int size[3];   // size = high - low + 1
   };

FFTManager
~~~~~~~~~~

.. code-block:: cpp

   // Forward r2c FFT. Applies 1/N^3 normalization.
   void Forward(const double* input, Kokkos::complex<double>* output);

   // Backward c2r FFT. Applies no normalization.
   void Backward(const Kokkos::complex<double>* input, double* output);

   // Returns the local Fourier-space box (global Fourier indices)
   Box3D fourier_space_box() const;

   // Returns the local real-space box (global cell indices)
   Box3D real_space_box() const;

   // Total number of points in the local Fourier/real space box
   std::size_t size_fourier_space_box() const;
   std::size_t size_real_space_box() const;

   // Returns a device-copyable helper for use in Kokkos kernels.
   // Capture by value in KOKKOS_LAMBDA.
   KernelHelper GetKernelHelper() const;

FFTManager::KernelHelper
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Flat index into the local Fourier-space array
   KOKKOS_INLINE_FUNCTION
   std::int64_t FourierFlatIndex(const int k, const int j, const int i) const;

   // Flat index into the local real-space array
   KOKKOS_INLINE_FUNCTION
   std::int64_t RealFlatIndex(const int k, const int j, const int i) const;

   // Integer wavevector components (handles negative frequency unwrapping).
   // For r2c transforms, kx >= 0 always.
   // Returns {kx, ky, kz}.
   KOKKOS_INLINE_FUNCTION
   std::array<int, 3> Wavevector(const int k, const int j, const int i) const;

UniformGridHelper
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Gather a single component of a named variable into a flat FFT-ready array.
   // output must be pre-allocated with size >= size_real_space_box()
   void GatherField(const std::string &var_name,
                    const int var_index,
                    parthenon::ParArray1D<Real> &output);

   // Scatter a flat array back to a named variable on the mesh.
   void ScatterField(const parthenon::ParArray1D<Real> &input,
                     const std::string &var_name,
                     const int var_index);

   // Returns a device-copyable helper for use in Kokkos kernels.
   // Capture by value in KOKKOS_LAMBDA.
   KernelHelper GetKernelHelper() const;

   // Local real-space box (global cell indices of this rank's domain)
   Box3D LocalMeshBox;

   // Per-meshblock box (interior cell bounds within a single meshblock)
   Box3D MeshBlockBox;

UniformGridHelper::KernelHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Flat index into the local real-space FFT array.
   // Call from within a par_for loop over blocks and interior cells.
   KOKKOS_INLINE_FUNCTION
   std::int64_t FlatIndex(int b, int k, int j, int i) const;

Limitations
-----------

* Only uniform grids are supported. AMR is not compatible with the current FFT infrastructure.
* ``num_packs = 1`` is required (one partition per rank).
* Only cubic domains are fully supported for physical wavenumber calculations.
  Non-cubic domains work for the FFT itself but wavenumber scaling must be handled manually.
* The r2c transform stores only modes with :math:`k_x \geq 0`, consistent with heFFTe's
  default convention.
* Currently only 3D transforms are supported.

See Also
--------

* :doc:`/src/interface/state` — registering variables for use with ``GatherField``/``ScatterField``
* `heFFTe documentation <https://icl-utk-edu.github.io/heffte/>`_
* Fourier transform example: ``example/fourier_transform/``