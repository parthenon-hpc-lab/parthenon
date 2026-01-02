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
    // Quantities needed for mapping data to/from meshblocks. Declared public for ease of access.
    std::array<std::int64_t, 3> local_loc_min{
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
    };
    std::array<std::int64_t, 3> local_loc_max{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min(),
    };
    int nx1b;
    int nx2b;
    int nx3b;
    std::int64_t nx1l;
    std::int64_t nx2l;
    std::int64_t nx3l;

  private:
    Mesh *mesh_;           
    bool initialized_ = false;
};

} // namespace parthenon
