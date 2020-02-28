//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef ATHENA_ARRAYS_HPP_
#define ATHENA_ARRAYS_HPP_
//! \file athena_arrays.hpp
//  \brief provides array classes valid in 1D to 5D.
//
//  The operator() is overloaded, e.g. elements of a 4D array of size [N4xN3xN2xN1]
//  are accessed as:  A(n,k,j,i) = A[i + N1*(j + N2*(k + N3*n))]
//  NOTE THE TRAILING INDEX INSIDE THE PARENTHESES IS INDEXED FASTEST

// C headers

// C++ headers
#include <cstddef>  // size_t
#include <cstring>  // memset()
#include <utility>  // swap(), transform()
#include <functional> // plus
#include <algorithm>
#include <assert.h>
#include <vector>

// Athena++ headers

template <typename T>
class AthenaArray {
 public:
  enum class DataStatus {empty, shallow_slice, allocated};  // formerly, "bool scopy_"

  // ctors
  // default ctor: simply set null AthenaArray
  AthenaArray() : pdata_(nullptr), nx1_(0), nx2_(0), nx3_(0),
                  nx4_(0), nx5_(0), nx6_(0), state_(DataStatus::empty) {}
  // ctor overloads: set expected size of unallocated container, maybe allocate (default)
  explicit AthenaArray(int nx1, DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(1), nx3_(1), nx4_(1), nx5_(1), nx6_(1),
      state_(init) { AllocateData(); }
  AthenaArray(int nx2, int nx1, DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(nx2), nx3_(1), nx4_(1), nx5_(1), nx6_(1),
      state_(init) { AllocateData(); }
  AthenaArray(int nx3, int nx2, int nx1, DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(1), nx5_(1), nx6_(1),
      state_(init) { AllocateData(); }
  AthenaArray(int nx4, int nx3, int nx2, int nx1, DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(1), nx6_(1),
      state_(init) { AllocateData(); }
  AthenaArray(int nx5, int nx4, int nx3, int nx2, int nx1,
              DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(nx5),  nx6_(1),
      state_(init) { AllocateData(); }
  AthenaArray(int nx6, int nx5, int nx4, int nx3, int nx2, int nx1,
              DataStatus init=DataStatus::allocated) :
      pdata_(nullptr), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(nx5), nx6_(nx6),
      state_(init) { AllocateData(); }
  // still allowing delayed-initialization (after constructor) via array.NewAthenaArray()
  // or array.InitWithShallowSlice() (only used in outputs.cpp + 3x other files)
  // TODO(felker): replace InitWithShallowSlice with ??? and remove shallow_copy enum val
  // TODO(felker): replace raw pointer with std::vector + reshape (if performance is same)

  // user-provided dtor, "rule of five" applies:
  ~AthenaArray();
  // define copy constructor and overload assignment operator so both do deep copies.
  __attribute__((nothrow)) AthenaArray(const AthenaArray<T>& t);
  __attribute__((nothrow)) AthenaArray<T> &operator= (const AthenaArray<T> &t);
  // define move constructor and overload assignment operator to transfer ownership
  __attribute__((nothrow)) AthenaArray(AthenaArray<T>&& t);
  __attribute__((nothrow)) AthenaArray<T> &operator= (AthenaArray<T> &&t);

  // public functions to allocate/deallocate memory for 1D-5D data
  __attribute__((nothrow)) void NewAthenaArray(int nx1);
  __attribute__((nothrow)) void NewAthenaArray(int nx2, int nx1);
  __attribute__((nothrow)) void NewAthenaArray(int nx3, int nx2, int nx1);
  __attribute__((nothrow)) void NewAthenaArray(int nx4, int nx3, int nx2, int nx1);
  __attribute__((nothrow)) void NewAthenaArray(int nx5, int nx4, int nx3, int nx2,
                                               int nx1);
  __attribute__((nothrow)) void NewAthenaArray(int nx6, int nx5, int nx4, int nx3,
                                               int nx2, int nx1);
  void DeleteAthenaArray();

  // public function to swap underlying data pointers of two equally-sized arrays
  void SwapAthenaArray(AthenaArray<T>& array2);
  void ZeroClear();

  // functions to get array dimensions
  int GetDim1() const { return nx1_; }
  int GetDim2() const { return nx2_; }
  int GetDim3() const { return nx3_; }
  int GetDim4() const { return nx4_; }
  int GetDim5() const { return nx5_; }
  int GetDim6() const { return nx6_; }
  int GetDim(size_t i) const {
    // TODO: remove if performance cirtical
    assert( 0 < i && i <= 6 && "AthenaArrays are max 6D" );
    switch (i) {
    case 1: return GetDim1();
    case 2: return GetDim2();
    case 3: return GetDim3();
    case 4: return GetDim4();
    case 5: return GetDim5();
    case 6: return GetDim6();
    }
    return -1;
  }

  std::vector<int> GetShape() const { return std::vector<int>({nx6_, nx5_, nx4_, nx3_, nx2_, nx1_}); }

  // a function to get the total size of the array
  int GetSize() const {
    if (state_ == DataStatus::empty)
      return 0;
    else
      return nx1_*nx2_*nx3_*nx4_*nx5_*nx6_;
  }
  std::size_t GetSizeInBytes() const {
    if (state_ == DataStatus::empty)
      return 0;
    else
      return nx1_*nx2_*nx3_*nx4_*nx5_*nx6_*sizeof(T);
  }

  size_t GetRank() const {
    for (int i = 6; i >= 1; i--) {
      if (GetDim(i) > 1) return i;
    }
    return 0;
  }

  bool IsShallowSlice() { return (state_ == DataStatus::shallow_slice); }
  bool IsEmpty() { return (state_ == DataStatus::empty); }
  bool IsAllocated() { return (state_ == DataStatus::allocated); }
  // "getter" function to access private data member
  // TODO(felker): Replace this unrestricted "getter" with a limited, safer alternative.
  // TODO(felker): Rename function. Conflicts with "AthenaArray<> data" OutputData member.
  T *data() { return pdata_; }
  const T *data() const { return pdata_; }
  T *begin() {return pdata_;}
  T *end()   {return pdata_+GetSize();}

  // overload "function call" operator() to access 1d-5d data
  // provides Fortran-like syntax for multidimensional arrays vs. "subscript" operator[]

  // "non-const variants" called for "AthenaArray<T>()" provide read/write access via
  // returning by reference, enabling assignment on returned l-value, e.g.: a(3) = 3.0;
  T &operator() (const int n) {
    int idx = n;
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  // "const variants" called for "const AthenaArray<T>" returns T by value, since T is
  // typically a built-in type (versus "const T &" to avoid copying for general types)
  T operator() (const int n) const {
    int idx = n;
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  T &operator() (const int n, const int i) {
    int idx = i + nx1_*n;
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  T operator() (const int n, const int i) const {
    int idx = i + nx1_*n;
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  T &operator() (const int n, const int j, const int i) {
    int idx = i + nx1_*(j + nx2_*n);
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  T operator() (const int n, const int j, const int i) const {
    int idx = i + nx1_*(j + nx2_*n);
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  T &operator() (const int n, const int k, const int j, const int i) {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*n));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  T operator() (const int n, const int k, const int j, const int i) const {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*n));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  T &operator() (const int m, const int n, const int k, const int j, const int i) {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*(n + nx4_*m)));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  T operator() (const int m, const int n, const int k, const int j, const int i) const {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*(n + nx4_*m)));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  // int l?, int o?
  T &operator() (const int p, const int m, const int n, const int k, const int j,
                 const int i) {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*(n + nx4_*(m + nx5_*p))));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }
  T operator() (const int p, const int m, const int n, const int k, const int j,
                const int i) const {
    int idx = i + nx1_*(j + nx2_*(k + nx3_*(n + nx4_*(m + nx5_*p))));
    assert((idx >= 0) && (idx < nx1_*nx2_*nx3_*nx4_*nx5_*nx6_));
    return pdata_[idx]; }

  AthenaArray<T> operator * (T scale) const {
    std::transform(pdata_, pdata_+GetSize(), pdata_, [scale](T val){return scale*val;});
  }

  AthenaArray<T>& operator*= (T scale) {
    std::transform(pdata_, pdata_+GetSize(), pdata_, [scale](T val){return scale*val;});
    return *this;
  }

  friend AthenaArray<T> operator- (const AthenaArray<T>& A) {
    AthenaArray<T> out = A;
    std::transform(A.pdata_, A.pdata_+A.GetSize(), out.pdata_, std::negate<T>() );
    return out;
  }

  AthenaArray<T>& operator+=(const AthenaArray<T>& other) {
    assert( GetSize() == other.GetSize() );
    std::transform(pdata_, pdata_+GetSize(), other.pdata_, pdata_, std::plus<T>() );
    return *this;
  }

  AthenaArray<T>& operator-=(const AthenaArray<T>& other) {
    assert( GetSize() == other.GetSize() );
    std::transform(pdata_, pdata_+GetSize(), other.pdata_, pdata_, std::minus<T>() );
    return *this;
  }

  friend AthenaArray<T> operator+ (const AthenaArray<T>& lhs, const AthenaArray<T>& rhs) {
    AthenaArray<T> out = lhs;
    out += rhs;
    return out;
  }

  friend AthenaArray<T> operator- (const AthenaArray<T>&  lhs, const AthenaArray<T>& rhs) {
    return lhs + -rhs;
  }

  // Checks that arrays point to same data with same shape
  // note this POINTER equivalence, not data equivalence
  bool operator== (const AthenaArray<T>& other) const;
  bool operator!= (const AthenaArray<T>& other) const { return !(*this == other); }

  void ShallowCopy(const AthenaArray<T> &src);
  // (deferred) initialize an array with slice from another array
  void InitWithShallowSlice(const AthenaArray<T> &src, const int dim,
			    const int indx, const int nvar);
  AthenaArray<T> slice(const int dim, const int indx, const int nvar) const;
  AthenaArray<T> slice(const int indx) const {return slice(GetRank(), indx, 1);}

 private:
  T *pdata_;
  int nx1_, nx2_, nx3_, nx4_, nx5_, nx6_;
  DataStatus state_;  // describe what "pdata_" points to and ownership of allocated data

  void AllocateData();
};

template<typename T>
AthenaArray<T> operator * (const T scale, const AthenaArray<T>& arr) { return arr*scale; }

// destructor

template<typename T>
AthenaArray<T>::~AthenaArray() {
  DeleteAthenaArray();
}

// copy constructor (does a deep copy)

template<typename T>
__attribute__((nothrow)) AthenaArray<T>::AthenaArray(const AthenaArray<T>& src) {
  nx1_ = src.nx1_;
  nx2_ = src.nx2_;
  nx3_ = src.nx3_;
  nx4_ = src.nx4_;
  nx5_ = src.nx5_;
  nx6_ = src.nx6_;
  if (src.pdata_) {
    std::size_t size = (src.nx1_)*(src.nx2_)*(src.nx3_)*(src.nx4_)*(src.nx5_)*(src.nx6_);
    pdata_ = new T[size]; // allocate memory for array data
    for (std::size_t i=0; i<size; ++i) {
      pdata_[i] = src.pdata_[i]; // copy data (not just addresses!) into new memory
    }
    state_ = DataStatus::allocated;
  }
}

// copy assignment operator (does a deep copy). Does not allocate memory for destination.
// THIS REQUIRES THAT THE DESTINATION ARRAY IS ALREADY ALLOCATED & THE SAME SIZE AS SOURCE

template<typename T>
__attribute__((nothrow))
AthenaArray<T> &AthenaArray<T>::operator= (const AthenaArray<T> &src) {
  if (this != &src) {
    // setting nxN_ is redundant given the above (unenforced) constraint on allowed usage
    nx1_ = src.nx1_;
    nx2_ = src.nx2_;
    nx3_ = src.nx3_;
    nx4_ = src.nx4_;
    nx5_ = src.nx5_;
    nx6_ = src.nx6_;
    std::size_t size = (src.nx1_)*(src.nx2_)*(src.nx3_)*(src.nx4_)*(src.nx5_)*(src.nx6_);
    for (std::size_t i=0; i<size; ++i) {
      this->pdata_[i] = src.pdata_[i]; // copy data (not just addresses!)
    }
    state_ = DataStatus::allocated;
  }
  return *this;
}

// move constructor
template<typename T>
__attribute__((nothrow)) AthenaArray<T>::AthenaArray(AthenaArray<T>&& src) {
  nx1_ = src.nx1_;
  nx2_ = src.nx2_;
  nx3_ = src.nx3_;
  nx4_ = src.nx4_;
  nx5_ = src.nx5_;
  nx6_ = src.nx6_;
  if (src.pdata_) {
    // && (src.state_ != DataStatus::allocated){  // (if forbidden to move shallow slices)
    //  ---- >state_ = DataStatus::allocated;

    // Allowing src shallow-sliced AthenaArray to serve as move constructor argument
    state_ = src.state_;
    pdata_ = src.pdata_;
    // remove ownership of data from src to prevent it from free'ing the resources
    src.pdata_ = nullptr;
    src.state_ = DataStatus::empty;
    src.nx1_ = 0;
    src.nx2_ = 0;
    src.nx3_ = 0;
    src.nx4_ = 0;
    src.nx5_ = 0;
    src.nx6_ = 0;
  }
}

// move assignment operator
template<typename T>
__attribute__((nothrow))
AthenaArray<T> &AthenaArray<T>::operator= (AthenaArray<T> &&src) {
  if (this != &src) {
    // free the target AthenaArray to prepare to receive src pdata_
    DeleteAthenaArray();
    if (src.pdata_) {
      nx1_ = src.nx1_;
      nx2_ = src.nx2_;
      nx3_ = src.nx3_;
      nx4_ = src.nx4_;
      nx5_ = src.nx5_;
      nx6_ = src.nx6_;
      state_ = src.state_;
      pdata_ = src.pdata_;

      src.pdata_ = nullptr;
      src.state_ = DataStatus::empty;
      src.nx1_ = 0;
      src.nx2_ = 0;
      src.nx3_ = 0;
      src.nx4_ = 0;
      src.nx5_ = 0;
      src.nx6_ = 0;
    }
  }
  return *this;
}

// Checks that arrays point to same data with same shape
// note this POINTER equivalence, not data equivalence
template<typename T>
bool AthenaArray<T>::operator== (const AthenaArray<T>& rhs) const {
  return (pdata_ == rhs.pdata_
	  && state_ == rhs.state_
	  && nx1_ == rhs.nx1_
	  && nx2_ == rhs.nx2_
	  && nx3_ == rhs.nx3_
	  && nx4_ == rhs.nx4_
	  && nx5_ == rhs.nx5_
	  && nx6_ == rhs.nx6_);
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::InitWithShallowSlice()
//  \brief shallow copy of nvar elements in dimension dim of an array, starting at
//  index=indx. Copies pointer to data, but not data itself.

//  Shallow slice is only able to address the "nvar" range in "dim", and all entries of
//  the src array for d<dim (cannot access any nx4=2, etc. entries if dim=3 for example)

template<typename T>
void AthenaArray<T>::InitWithShallowSlice(const AthenaArray<T> &src, const int dim,
                                          const int indx, const int nvar) {
  pdata_ = src.pdata_;
  if (dim == 6) {
    nx6_ = nvar;
    nx5_ = src.nx5_;
    nx4_ = src.nx4_;
    nx3_ = src.nx3_;
    nx2_ = src.nx2_;
    nx1_ = src.nx1_;
    pdata_ += indx*(nx1_*nx2_*nx3_*nx4_*nx5_);
  } else if (dim == 5) {
    nx6_ = 1;
    nx5_ = nvar;
    nx4_ = src.nx4_;
    nx3_ = src.nx3_;
    nx2_ = src.nx2_;
    nx1_ = src.nx1_;
    pdata_ += indx*(nx1_*nx2_*nx3_*nx4_);
  } else if (dim == 4) {
    nx6_ = 1;
    nx5_ = 1;
    nx4_ = nvar;
    nx3_ = src.nx3_;
    nx2_ = src.nx2_;
    nx1_ = src.nx1_;
    pdata_ += indx*(nx1_*nx2_*nx3_);
  } else if (dim == 3) {
    nx6_ = 1;
    nx5_ = 1;
    nx4_ = 1;
    nx3_ = nvar;
    nx2_ = src.nx2_;
    nx1_ = src.nx1_;
    pdata_ += indx*(nx1_*nx2_);
  } else if (dim == 2) {
    nx6_ = 1;
    nx5_ = 1;
    nx4_ = 1;
    nx3_ = 1;
    nx2_ = nvar;
    nx1_ = src.nx1_;
    pdata_ += indx*(nx1_);
  } else if (dim == 1) {
    nx6_ = 1;
    nx5_ = 1;
    nx4_ = 1;
    nx3_ = 1;
    nx2_ = 1;
    nx1_ = nvar;
    pdata_ += indx;
  }
  state_ = DataStatus::shallow_slice;
  return;
}

template<typename T>
void AthenaArray<T>::ShallowCopy(const AthenaArray<T> &src) {
  pdata_ = src.pdata_;
  nx6_ = src.nx6_;
  nx5_ = src.nx5_;
  nx4_ = src.nx4_;
  nx3_ = src.nx3_;
  nx2_ = src.nx2_;
  nx1_ = src.nx1_;
  state_ = DataStatus::shallow_slice;
  return;
}

template<typename T>
AthenaArray<T> AthenaArray<T>::slice(const int dim, const int indx, const int nvar) const {
  AthenaArray<T> out;
  out.InitWithShallowSlice(*this, dim, indx, nvar);
  return out;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief allocate new 1D array with elements initialized to zero.

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = 1;
  nx3_ = 1;
  nx4_ = 1;
  nx5_ = 1;
  nx6_ = 1;
  pdata_ = new T[nx1](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 2d data allocation

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx2, int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = nx2;
  nx3_ = 1;
  nx4_ = 1;
  nx5_ = 1;
  nx6_ = 1;
  pdata_ = new T[nx1*nx2](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 3d data allocation

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx3, int nx2, int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = nx2;
  nx3_ = nx3;
  nx4_ = 1;
  nx5_ = 1;
  nx6_ = 1;
  pdata_ = new T[nx1*nx2*nx3](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 4d data allocation

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx4, int nx3, int nx2,
                                                             int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = nx2;
  nx3_ = nx3;
  nx4_ = nx4;
  nx5_ = 1;
  nx6_ = 1;
  pdata_ = new T[nx1*nx2*nx3*nx4](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 5d data allocation

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx5, int nx4, int nx3,
                                                             int nx2, int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = nx2;
  nx3_ = nx3;
  nx4_ = nx4;
  nx5_ = nx5;
  nx6_ = 1;
  pdata_ = new T[nx1*nx2*nx3*nx4*nx5](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 6d data allocation

template<typename T>
__attribute__((nothrow)) void AthenaArray<T>::NewAthenaArray(int nx6, int nx5, int nx4,
                                                             int nx3, int nx2, int nx1) {
  state_ = DataStatus::allocated;
  nx1_ = nx1;
  nx2_ = nx2;
  nx3_ = nx3;
  nx4_ = nx4;
  nx5_ = nx5;
  nx6_ = nx6;
  pdata_ = new T[nx1*nx2*nx3*nx4*nx5*nx6](); // allocate memory and initialize to zero
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::DeleteAthenaArray()
//  \brief  free memory allocated for data array

template<typename T>
void AthenaArray<T>::DeleteAthenaArray() {
  // state_ is tracked partly for correctness of delete[] operation in DeleteAthenaArray()
  switch (state_) {
    case DataStatus::empty:
    case DataStatus::shallow_slice:
      pdata_ = nullptr;
      break;
    case DataStatus::allocated:
      delete[] pdata_;
      pdata_ = nullptr;
      state_ = DataStatus::empty;
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::SwapAthenaArray()
//  \brief  swap pdata_ pointers of two equally sized AthenaArrays (shallow swap)
// Does not allocate memory for either AthenArray
// THIS REQUIRES THAT THE DESTINATION AND SOURCE ARRAYS BE ALREADY ALLOCATED (state_ !=
// empty) AND HAVE THE SAME SIZES (does not explicitly check either condition)

template<typename T>
void AthenaArray<T>::SwapAthenaArray(AthenaArray<T>& array2) {
  std::swap(pdata_, array2.pdata_);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::ZeroClear()
//  \brief  fill the array with zero

template<typename T>
void AthenaArray<T>::ZeroClear() {
  switch (state_) {
    case DataStatus::empty:
      break;
    case DataStatus::shallow_slice:
    case DataStatus::allocated:
      // allocate memory and initialize to zero
      std::memset(pdata_, 0, GetSizeInBytes());
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::AllocateData()
//  \brief  to be called in non-default ctors, if immediate memory allocation is requested
//          (could replace all "new[]" calls in NewAthenaArray function overloads)

template<typename T>
void AthenaArray<T>::AllocateData() {
  switch (state_) {
    case DataStatus::empty:
    case DataStatus::shallow_slice: // init=shallow_slice should never be passed to ctor
      break;
    case DataStatus::allocated:
      // allocate memory and initialize to zero
      pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
      break;
  }
}

#endif // ATHENA_ARRAYS_HPP_
