#pragma once

#include <memory>
#include <complex>
#include "parthenon_arrays.hpp"

namespace parthenon {

class Mesh;

class FFTManager {
friend class Mesh;

public:
  explicit FFTManager(Mesh *mesh);
  ~FFTManager();

  void Initialize();

  void Forward(const ParArray1D<double> &input,
               ParArray1D<std::complex<double>> &output);

  void Backward(const ParArray1D<std::complex<double>> &input,
                ParArray1D<double> &output);
  // -----------------------------
  // Box info 
  // -----------------------------
  struct Box3D {
      int low[3];
      int high[3];
      int size[3];   // size in each dimension: high - low + 1
  };

  Box3D fourier_space_box() const;   
  Box3D real_space_box() const;      

  std::size_t size_fourier_space_box() const;  // total number of points
  std::size_t size_real_space_box() const;

private:
  struct Impl;                 // opaque implementation
  std::unique_ptr<Impl> impl_; // owns backend-specific data

  Mesh *mesh_;
  bool initialized_ = false;
};

} // namespace parthenon