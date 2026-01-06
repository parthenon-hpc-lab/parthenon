#pragma once

#include <memory>
#include "heffte.h"

namespace parthenon {
class Mesh;
class FFTManager {
  
  friend class Mesh;
  
  public:

    explicit FFTManager(Mesh *mesh);
    void Initialize();
    auto Forward(int field);
    auto Backward(const std::vector<std::complex<double>> &input);

    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::default_backend<heffte::tag::cpu>::type>> fft_plan_;

  private:
    Mesh *mesh_;           
    bool initialized_ = false;
};

} // namespace parthenon
