//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef UTILS_NAN_PAYLOAD_TAG_HPP_
#define UTILS_NAN_PAYLOAD_TAG_HPP_ 

#include <cassert> 
#include <cmath>
#include <limits>
#include <numeric>
#include <string.h>
#include <type_traits>

#include "utils/concepts_lite.hpp"

namespace parthenon {

namespace impl {

template <int NBYTES>
struct contiguous_bitset {
  contiguous_bitset() = default;
  contiguous_bitset(char val) : bytes{val} {}
  char bytes[NBYTES];

  static constexpr int char_bit_size = sizeof('a') * 8;  
  
  void SetOne(const std::size_t idx) { 
    char& byte = bytes[idx / char_bit_size];
    byte = byte | (1 << (idx % char_bit_size));
  }

  void SetZero(const std::size_t idx) { 
    char& byte = bytes[idx / char_bit_size];
    byte = byte & ~(1 << (idx % char_bit_size));
  }

  void Flip(const std::size_t idx) { 
    char& byte = bytes[idx / char_bit_size];
    byte = byte ^ (1 << (idx % char_bit_size));
  }

  template<class T>
  void SetEndBytes(T val) { 
    static_assert(sizeof(val) <= NBYTES * sizeof(std::declval<char>()), 
        "Input type is too large for given contiguous_bitset.");
    memcpy(bytes, &val, sizeof(val));
  }
};
}

template<class T, class U, typename std::enable_if<sizeof(T) == sizeof(U), int>::type = 0>
bool BitwiseCompare(const T& a, const U& b) {
  // memcmp returns zero if the memory is the same 
  // which is a little confusing
  return !memcmp(&a, &b, sizeof(a));
}

template<class T, REQUIRES(std::numeric_limits<T>::is_iec559)> 
int GetNaNTag(T val) {
    uint8_t tag; 
    memcpy(&tag, &val, sizeof(tag));
    return tag; 
}

template<class T, REQUIRES(!std::numeric_limits<T>::is_iec559)> 
int GetNaNTag(T val) {
    return -1;
}

template<class T, REQUIRES(std::numeric_limits<T>::is_iec559)>
T GetNaNWithPayloadTag(uint8_t tag = 1) { 
    double flag_nan = std::numeric_limits<T>::quiet_NaN();
    auto& flag_bits = 
        reinterpret_cast<impl::contiguous_bitset<sizeof(flag_nan)>&>(flag_nan);

    // val must be > 0 otherwise since val = 0 just keeps the value of 
    // the default quiet NaN 
    assert(val > 0);
    flag_bits.SetEndBytes(tag);

    // Do a few quick checks to make sure there isn't anything 
    // weird going on
    assert(std::isnan(flag_nan));
    assert(!BitwiseCompare(std::numeric_limits<T>::signaling_NaN(), flag_nan));
    assert(!BitwiseCompare(std::numeric_limits<T>::quiet_NaN(), flag_nan));

    return flag_nan; 
}

template<class T, REQUIRES(!std::numeric_limits<T>::is_iec559)>
T GetNaNWithPayloadTag(uint8_t tag = 1) { 
    // TODO (LFR): Probably need to warn here that we can't tag without IEEE
    return std::numeric_limits<T>::quiet_NaN();
}

/*
template<class T> 
class NaNPayloadTag {
 public:
  NaNPayloadTag(uint8_t tag = 1) {
    static_assert(std::numeric_limits<T>::is_iec559, 
       "T does not conform to the IEEE standard, so we "
       "can't safely hide flags in the NaN payload.");
    
    flag_ = std::numeric_limits<T>::quiet_NaN();
    auto& flag_bits = 
        reinterpret_cast<impl::contiguous_bitset<sizeof(flag_)>&>(flag_);

    // val must be > 0 otherwise since val = 0 just keeps the value of 
    // the default quiet NaN 
    assert(val > 0);
    flag_bits.SetEndBytes(tag);

    // Do a few quick checks to make sure there isn't anything 
    // weird going on
    assert(std::isnan(flag_));
    assert(!BitwiseCompare(std::numeric_limits<T>::signaling_NaN(), flag_));
    assert(!BitwiseCompare(std::numeric_limits<T>::quiet_NaN(), flag_));
  }
  
  T val() {return flag_;}
  operator T() {return flag_;}

  int flag() {
    uint8_t flag_val; 
    memcpy(&flag_val, &flag_, sizeof(flag_val));
    return flag_val; 
  }
  
 protected:
  T flag_; 
};
*/

} // namespace parthenon

#endif // UTILS_NAN_PAYLOAD_TAG_HPP_