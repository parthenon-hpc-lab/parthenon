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
`PARARRAY_TEMP` macro. It is provided to name the array based on the
file and line where it is created.

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

You can slice a `ParArrayND` using `ParArrayND.Slice(Args...args)`. It
takes the same arguments for slicing as `Kokkos::subview`.

You can also slice with a syntax closer to `AthenArray`'s
`InitWithShallowSlice`. It's called `SliceD`. It is templated on the
dimension into which you want to slice. Then it can take a `std::pair`
of the slice range `start:finish` or it can take two integgers,
specifying the index and the number of elements in that dimension to
include. E.g.,
 ```C+
// this is equivalent to
// b.InitWithShallowSlice(a, dim, indx, size);
// for athena_arrays
auto b = a.SliceD<dim>(indx,size);
```
You should *always* use `auto` when extracting slices from
`ParArrayND` as the underlying, templated type may change. The same is
true for `Get`, and `GetMirror`, which are described below.

### Get

A `Kokkos::View` with the dimensionality you want can be extracted
with `ParArrayND.Get<D>();`. This returns a rank-D view. Dimensions
higher than D are set to zero.

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
In addition, `GetHostMirrorAndCopy()` creates a new and deep copies the content, e.g.,
```C++
auto my_host_array = my_array.getHostMirrorAndCopy();
```

### A note on templates

Strictly, `ParArrayND` is a specialization of `ParArrayNDGeneric`,
which wraps an arbitrary container. The specializations and type
aliases available are as follows:

```C++
template<typename T, typename Layout=LayoutWrapper>
using device_view_t = Kokkos::View<T******,Layout,DevMemSpace>;

template<typename T, typename Layout=LayoutWrapper>
using host_view_t = typename device_view_t<T,Layout>::HostMirror;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayND = ParArrayNDGeneric<device_view_t<T,Layout>>;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayHost = ParArrayNDGeneric<host_view_t<T,Layout>>;
```

### Examples of use

See the [unit test](../tst/unit/test_pararrays.cpp) for how to use `ParArrayND`.
