//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// AI was used to assist with NCCL error handling and CUDA event management.
//========================================================================================
#ifndef BVALS_COMMS_NCCL_UTILS_HPP_
#define BVALS_COMMS_NCCL_UTILS_HPP_

#include <sstream>
#include <string>

#include "config.hpp"
#include "utils/error_checking.hpp"

#ifdef ENABLE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>

namespace parthenon {

inline void CheckNCCL(ncclResult_t result, const char *expression, const char *file,
                      int line) {
  if (result == ncclSuccess) return;
  std::stringstream msg;
  msg << "NCCL call failed at " << file << ":" << line << ": " << expression << ": "
      << ncclGetErrorString(result);
  PARTHENON_FAIL(msg.str());
}

inline void CheckCudaEvent(cudaError_t result, const char *expression, const char *file,
                           int line) {
  if (result == cudaSuccess) return;
  std::stringstream msg;
  msg << "CUDA event call failed at " << file << ":" << line << ": " << expression << ": "
      << cudaGetErrorString(result);
  PARTHENON_FAIL(msg.str());
}

#define PARTHENON_NCCL_CHECK(call)                                                       \
  ::parthenon::CheckNCCL((call), #call, __FILE__, __LINE__)
#define PARTHENON_CUDA_EVENT_CHECK(call)                                                 \
  ::parthenon::CheckCudaEvent((call), #call, __FILE__, __LINE__)

class CudaEvent {
 public:
  CudaEvent() {
    PARTHENON_CUDA_EVENT_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }
  ~CudaEvent() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  CudaEvent(CudaEvent &&other) noexcept : event_(other.event_) { other.event_ = nullptr; }
  CudaEvent &operator=(CudaEvent &&other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  void Record(cudaStream_t stream) {
    PARTHENON_CUDA_EVENT_CHECK(cudaEventRecord(event_, stream));
  }
  void Wait(cudaStream_t stream) {
    PARTHENON_CUDA_EVENT_CHECK(cudaStreamWaitEvent(stream, event_, 0));
  }
  bool Query() {
    const auto result = cudaEventQuery(event_);
    if (result == cudaSuccess) return true;
    if (result == cudaErrorNotReady) return false;
    PARTHENON_CUDA_EVENT_CHECK(result);
    return false;
  }

 private:
  cudaEvent_t event_{nullptr};
};

} // namespace parthenon
#endif // ENABLE_NCCL

#endif // BVALS_COMMS_NCCL_UTILS_HPP_
