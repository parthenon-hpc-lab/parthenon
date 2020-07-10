# Parthenon Built-in Physical Constants

PhysicalConstants is a class that provides numerical values of physical
constants in common unit systems (SI and CGS). This is provided in the hopes
that parthenon-derived applications can conveniently share exactly matching
values for these constants, to avoid bugs that can be associated with
disagreement and the effort required to synchronize a consistent move to newer
values of physical constants.

Internal values are hardcoded in SI units, and specific unit systems are
realized by providing a struct of conversion factors as a template parameter.

PhysicalConstants is purely constexpr, and the internal unit conversion factors
are protected members to allow for custom constants classes to derive from this
class.

Both verbose and terse names are provided for each constant as public (albeit
constexpr) members.

### Usage

To construct, call this class with PhysicalConstants<UNITSYSTEM>(), where
UNITSYSTEM is a struct of conversion factors from SI units; both SI and CGS unit
systems are provided in the parthenon::constants namespace. Then, to access
constants simply use the public data members.
