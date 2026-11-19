Utilities
=========

``TypeList``\ s
---------------

Provides a wrapper class around a variadic pack of types to simplify
performing compile time operations on the pack. There are templated 
types defined giving the type at a particular index in the pack, types 
for extracting sub-``TypeList``\ s of the original type list, and ``constexpr``
functions for getting the index of the first instance of a type in the 
pack. Additionally it provides a capability for iterating an ``auto`` lambda
over the type list, which can be useful for calling a ``static`` function 
defined for each of the types on each of the types. *In the future, it
would be nice to have the capability to make a unique type list from
another type list (i.e. the unique one only a single instance of each type
in the original type list)*

``TypeList``\ s have many applications and are commonly found in many 
codebases, but in Parthenon one of the main use cases is for storing 
lists of types associated with field variables that are used in type 
based ``SparsePack``\ s.  

Robust
------

Provides a number of functions for doing operations on floating point
numbers that are bounded, avoid division by zero, etc. 

C++11 Style Concepts Implementation
-----------------------------------

*This documentation needs to be written (see issue #695), but there are
extensive comments in src/utlils/concepts_lite.hpp and examples of
useage in tst/unit/test_concepts_lite.hpp*

``Indexer``
-----------

Provides functionality for iterating over an arbitrary dimensional
hyper-rectangular index space using a flattened loop. Specific 
instantiations, e.g. ``Indexer5D``, are provided up to eight 
dimensions. Useage:

.. code:: cpp

  Indexer4D idxer({0, 3}, {1, 2}, {0, 5}, {10, 16});
  for (int flat_idx = 0; flat_idx < idxer.size(); ++flat_idx) {
    auto [i, j, k, l] = idxer(flat_idx);
    // Do stuff in the 4D index space...
  }
