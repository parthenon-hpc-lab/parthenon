Fourier Transforms
==================

Parthenon provides infrastructure for performing distributed Fast Fourier Transforms (FFTs)
on uniform meshes via the :cpp:class:`FFTManager` and :cpp:class:`UniformGridHelper` classes.
These are built on top of `heFFTe <https://github.com/icl-utk-edu/heffte>`_ and support
both CPU and GPU backends transparently.

.. note::
   FFT functionality requires ``pack_size = -1`` in the input file, meaning all meshblocks
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

   auto fftManager       = pmesh->GetFFTManager();
   auto uniformGridHelper = pmesh->GetUniformGridHelper();

Normalization Convention
------------------------

The forward transform applies a :math:`1/N^3` normalization, and the backward transform
applies no normalization. This means the round-trip (forward followed by backward) recovers
the original field exactly, and Parseval's theorem reads:

.. math::

   \sum_{\mathbf{k}} |\hat{f}(\mathbf{k})|^2 = \frac{1}{N^3} \sum_{\mathbf{x}} |f(\mathbf{x})|^2

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

   parthenon::ParArray1D<Real>                  input("input",  fft_size_inbox);
   parthenon::ParArray1D<std::complex<Real>>    output("output", fft_size_outbox);
   parthenon::ParArray1D<Real>                  result("result", fft_size_inbox);

Gathering a field from the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`UniformGridHelper` provides :cpp:func:`GatherField` to copy a single component
of a named Parthenon variable into a flat array suitable for FFT input:

.. code-block:: cpp

   // Gather the x-component of the magnetic field (component index IB1)
   uniformGridHelper->GatherField("cons", IB1, input);

For derived quantities that require computation (e.g. velocity :math:`u = m/\rho`),
use :cpp:func:`FFTFlatIndex` to write a custom gather loop:

.. code-block:: cpp

   auto &md   = pmesh->mesh_data.Get();
   auto  cons = md->PackVariables(std::vector<std::string>{"cons"});

   IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
   IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
   IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

   parthenon::par_for(
       "GatherVelocity", 0, pmesh->GetNumMeshBlocksThisRank() - 1,
       kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
       KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
           const auto idx = uniformGridHelper->FFTFlatIndex(b, k, j, i);
           const auto rho = cons(b, IDN, k, j, i);
           input(idx) = cons(b, IVX, k, j, i) / rho;
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

For vector fields, loop over components:

.. code-block:: cpp

   std::array<parthenon::ParArray1D<Real>, 3>               B;
   std::array<parthenon::ParArray1D<std::complex<Real>>, 3> B_hat;
   const std::array<int, 3> B_indices = {IB1, IB2, IB3};

   for (int i = 0; i < 3; i++) {
       B[i]     = parthenon::ParArray1D<Real>("B",     fft_size_inbox);
       B_hat[i] = parthenon::ParArray1D<std::complex<Real>>("B_hat", fft_size_outbox);
       uniformGridHelper->GatherField("cons", B_indices[i], B[i]);
       fftManager->Forward(B[i].data(), B_hat[i].data());
   }

Processing in Fourier space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   ``std::complex<Real>`` arithmetic is not supported inside GPU kernels. When accessing
   complex arrays inside a ``par_for`` kernel, cast to ``Kokkos::complex<Real>`` first:

   .. code-block:: cpp

      auto output_kk = reinterpret_cast<Kokkos::complex<Real>*>(output.data());

      parthenon::par_for(
          "FourierSpaceKernel", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int kz_idx, const int ky_idx, const int kx_idx) {
              const std::int64_t idx = ...;
              // use output_kk[idx] instead of output(idx)
              auto val = output_kk[idx] * some_kokkos_complex;
          });

   ``Kokkos::complex`` and ``std::complex`` have identical memory layouts, so the
   reinterpret cast is safe. The cast must be done **before** the lambda capture —
   capturing a ``ParArray1D<std::complex<Real>>`` and calling ``.data()`` inside
   the kernel will not work on GPU.

The local Fourier space box — the subset of Fourier modes owned by this rank — is
accessible via :cpp:func:`FFTManager::fourier_space_box`:

.. code-block:: cpp

   auto outbox = fftManager->fourier_space_box();

   IndexRange ib, jb, kb;
   ib.s = outbox.low[0]; ib.e = outbox.high[0];
   jb.s = outbox.low[1]; jb.e = outbox.high[1];
   kb.s = outbox.low[2]; kb.e = outbox.high[2];

   const auto Nx = uniformGridHelper->global_mesh_size[0];
   const auto Ny = uniformGridHelper->global_mesh_size[1];
   const auto Nz = uniformGridHelper->global_mesh_size[2];

Inside a ``par_for`` over the Fourier box, the flat index and physical wavenumbers
are computed as:

.. code-block:: cpp

   parthenon::par_for(
       "FourierSpaceKernel", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
       KOKKOS_LAMBDA(const int kz_idx, const int ky_idx, const int kx_idx) {

           // unwrap negative frequencies (r2c: kx >= 0 always)
           const auto kz = kz_idx <= Nz/2 ? kz_idx : kz_idx - Nz;
           const auto ky = ky_idx <= Ny/2 ? ky_idx : ky_idx - Ny;
           const auto kx = kx_idx;

           // physical wavenumbers (assuming cubic box of side L)
           const Real kx_phys = 2.0 * M_PI * kx / L;
           const Real ky_phys = 2.0 * M_PI * ky / L;
           const Real kz_phys = 2.0 * M_PI * kz / L;

           // flat index into the local Fourier box
           const std::int64_t idx =
               ((std::int64_t)(kz_idx - kb.s) * (jb.e - jb.s + 1) + (ky_idx - jb.s))
               * (ib.e - ib.s + 1) + kx_idx - ib.s;

           // ... process output[idx] ...
       });

.. note::
   The r2c transform only stores modes with :math:`k_x \geq 0`. Hermitian symmetry
   must be accounted for when computing quantities like the power spectrum — modes with
   :math:`0 < k_x < N_x/2` contribute twice:

   .. code-block:: cpp

      const auto fac = ((kx > 0) && (2 * kx != Nx)) ? 2.0 : 1.0;

Scattering a field back to the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:func:`UniformGridHelper::ScatterField` copies a flat array back to a named
Parthenon variable on the mesh:

.. code-block:: cpp

   parthenon::ParArray1D<Real> result("result", fft_size_inbox);
   // ... fill result ...
   uniformGridHelper->ScatterField(result, "my_derived_field", 0);

The variable must be registered in the package before use (see :ref:`state`).

Power Spectrum
--------------

The generic :cpp:class:`SpectralOutput` output type computes the isotropically binned
power spectrum of any named Parthenon variable. Configure it in the input file:

.. code-block:: ini

   <output1>
   file_type    = spectrum
   variable     = cons
   components   = 5 6 7
   output_label = B
   dt           = 0.1

This computes :math:`E(k) = \sum_{|\mathbf{k}'| \in \mathrm{bin}} |\hat{f}(\mathbf{k}')|^2`
and writes it to a text file with columns ``bin``, ``E_sum``, ``k_sum``, ``count``.

The output label is used in the filename. Multiple components are summed in quadrature.

API Reference
-------------

FFTManager
~~~~~~~~~~

.. code-block:: cpp

   // Forward r2c FFT. Applies 1/N^3 normalization.
   void Forward(const double* input, std::complex<double>* output);

   // Backward c2r FFT. Applies no normalization.
   void Backward(const std::complex<double>* input, double* output);

   // Returns the local Fourier space box (global indices)
   Box3D fourier_space_box() const;

   // Returns the local real space box (global indices)
   Box3D real_space_box() const;

   // Total number of points in the local Fourier/real space box
   std::size_t size_fourier_space_box() const;
   std::size_t size_real_space_box() const;

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

   // Device-callable flat index for use inside par_for kernels.
   // Maps (block, k, j, i) to a flat index into the local real-space FFT array.
   KOKKOS_INLINE_FUNCTION
   std::int64_t FFTFlatIndex(int b, int k, int j, int i) const;

   // Global mesh dimensions
   std::array<int, 3> global_mesh_size;

   // Local mesh dimensions on this rank
   std::array<int, 3> local_mesh_size;

Limitations
-----------

* Only uniform grids are supported. AMR (adaptive mesh refinement) is not compatible
  with the current FFT infrastructure.
* ``pack_size = -1`` is required (one partition per rank).
* Only cubic domains are supported for wavenumber computations involving physical
  units. Non-cubic domains work for the FFT itself but physical wavenumber
  calculations must be handled manually.
* The r2c transform assumes the x-direction is the transform direction with
  :math:`k_x \geq 0`, consistent with heFFTe's default convention.

See Also
--------

* :doc:`/src/interface/state` — registering variables for use with ``GatherField``/``ScatterField``
* `heFFTe documentation <https://icl-utk-edu.github.io/heffte/>`_
* Fourier transform example: ``example/fourier_transform/``