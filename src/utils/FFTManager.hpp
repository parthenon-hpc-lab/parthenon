#pragma once

#include <memory>
#include "heffte.h"

namespace parthenon {
class Mesh;
}

namespace parthenon {

class FFTManager {
  friend class Mesh;
  public:
    explicit FFTManager(Mesh *mesh);

    void Initialize();
    auto Forward(const std::vector<double> &input) {
        Initialize();
        return fft_plan_->forward(input, heffte::scale::full);
    }

    auto Backward(const std::vector<std::complex<double>> &input) {
        Initialize();
        return fft_plan_->backward(input, heffte::scale::full);
    }
  
  std::unique_ptr<heffte::fft3d_r2c<heffte::backend::default_backend<heffte::tag::cpu>::type>> fft_plan_;

  private:
    Mesh *mesh_;           
    bool initialized_ = false;
};

} // namespace parthenon
