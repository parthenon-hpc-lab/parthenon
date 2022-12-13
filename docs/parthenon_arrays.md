# Parthenon Set-Dimensional Arrays

We provide type aliases to `Kokkos::View`. They are

```C++
template <typename T>
using ParArray1D = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray2D = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray3D = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray4D = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray6D = Kokkos::View<T ******, LayoutWrapper, DevMemSpace>;
```
where `LayoutWrapper` is currently hardcoded to `Kokkos::LayoutRight`.
`DevMemSpace` is the memory space associated with the default execution space.
If UVM is enabled, it is `Kokkos::CudaUVMSpace`.

# Parthenon Arbitrary-Dimensional Arrays

`ParArrayND` is a wrapper around a rank 6 `Kokkos::View`. It provides
a rank-agnostic way to create, manage, and carry
`Kokkos::Views`. Since it is built on `Kokkos::View`, it is reference
counted, works on GPUs, and is almost as performant as `Kokkos::View`.

The index and size convention is the same as in `Athena++`. The
fastest moving indexes come last.

### Constructors

To construct, call it with `ParArrayND(label,dimensions...)`, where
dimensions is some number of ints specifying the shape of the
array. e.g.,
```C++
ParArrayND<Real> myArray("a 3d array", 6, 5, 4);
```

If you don't know what to name your array, try using the
`PARARRAY_TEMP` macro as the name. It is provided to name the array
based on the file and line where it is created.

### Rank counting

The rank of the object is indexed from 1, not 0. In other words, the
code
```C++
ParArrayND<Real> myArray("a 3d array", 6, 5, 4);
std::cout << myArray.GetDim(3) << " "
          << myArray.GetDim(2) << " "
          << myArray.GetDim(1) << std::endl;
```
prints out `6 5 4`.

### Accessors

Similarly, you can access it using `operator()` with up to six
integers specifying the indices. Unspecified (slow-moving) indices are
assumed to be zero.

### Slicing

You can slice a `ParArrayND` using `Kokkos::subview` as described
in Kokkos documentation. We have overloaded `Kokkos::subview`
to support this.

### Get

A `Kokkos::View` with the dimensionality you want can be extracted
with `ParArrayND.Get<D>();`. This returns a rank-D view. Dimensions
higher than D are set to zero.

#### Type Subtleties

Note that the type returned by `ParArrayND.Get<D>()` is not not a
simple type. For example, the type returned by `ParArrayND<Real>.Get<4>();` is:
```C++
Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<0> >
```
while the type for `ParArray4D<Real>` is
```C++
Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::HostSpace>
```
These two types are compatible. So you can set one equal to the other and do implicit casts.
I.e., the following works:
```C++
void doNothingByValue(ParArray4D<Real> array) {}
ParArrayND<Real> ndarray;
auto b = ndarray.Get<4>();
doNothingByValue(b);
```
However, implicit casts on reference variables are not performed. So the following fails:
```C++
void doNothingByReference(ParArray4D<Real>& array) {}
ParArrayND<Real> ndarray;
auto b = ndarray.Get<4>();
doNothingByReference(b);
```
To avoid this issue, you can:
- Explicitly typecast `ParArrayND` when using it in conjunction with Kokkos views.
- Pass views by reference
- Template appropriate functions on array type

For more details, see [here](https://github.com/lanl/parthenon/issues/143).

### Mirrors and Deep Copies

`ParArrayND` requires mirrors and deep copies, just like the `Kokkos`
views it wraps. You can get one via, e.g.,
```C++
auto my_mirror = my_array.GetMirror(my_memory_space());
```
 you can then deep copy into `my_mirror` with
```C++
my_mirror.DeepCopy(my_array);
```
`ParArrayND` provides two convenience functions, `GetHostMirror()` and
`GetDeviceMirror()` which put a mirror on the host and device
respectively.
In addition, `GetHostMirrorAndCopy()` creates a new `ParArrayND` on the host
with identical layout and deep copies the content, e.g.,
```C++
auto my_host_array = my_array.getHostMirrorAndCopy();
```

### A note on templates

Strictly, `ParArrayND` is a specialization of `ParArrayGeneric`,
which wraps an arbitrary container. The specializations and type
aliases available are as follows:

```C++
template<typename T, typename Layout=LayoutWrapper>
using device_view_t = Kokkos::View<T******,Layout,DevMemSpace>;

template<typename T, typename Layout=LayoutWrapper>
using host_view_t = typename device_view_t<T,Layout>::HostMirror;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayND = ParArrayGeneric<device_view_t<T,Layout>>;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayHost = ParArrayGeneric<host_view_t<T,Layout>>;
```

### Examples of use

See the [unit test](../tst/unit/test_pararrays.cpp) for how to use `ParArrayND`.
