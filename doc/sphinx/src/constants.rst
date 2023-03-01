Parthenon Built-in Physical Constants
=====================================

PhysicalConstants is a class that provides numerical values of physical
constants in common unit systems (SI and CGS). This is provided in the
hopes that parthenon-derived applications can conveniently share exactly
matching values for these constants, to avoid bugs that can be
associated with disagreement and the effort required to synchronize a
consistent move to newer values of physical constants.

Internal values are hardcoded in SI units, and specific unit systems are
realized by providing a struct of conversion factors as a template
parameter.

PhysicalConstants is purely constexpr, and the internal unit conversion
factors are protected members to allow for custom constants classes to
derive from this class.

Both verbose and terse names are provided for each constant as ``public``
(albeit ``constexpr``) members.

Usage
-----

To construct, call this class with ``PhysicalConstants()``, where ``UNITSYSTEM``
is a struct of conversion factors from SI units; both SI and CGS unit
systems are provided in the ``parthenon::constants`` namespace. Then, to
access constants simply use the public data members.

Defining a new unit system
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom unit system, create a struct of (ideally constexpr)
unit system conversion factors from SI to be used by ``PhysicalConstants``
as a template parameter, e.g.

.. code:: c++

   struct SIButKeVTemperatures {
     static constexpr double length = 1.;      // meter
     static constexpr double mass = 1.;        // kilogram
     static constexpr double time = 1.;        // second
     static constexpr double temperature = 8.6173e-8; // keV
     static constexpr double current = 1.;     // Amp
     static constexpr double charge = 1.;      // Coulomb
     static constexpr double capacitance = 1.; // Farad
     static constexpr double angle = 1.;       // Radian
   };

Geting values of physical constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get values of constants, create a ``PhysicalConstants`` class with the
appropriate unit system as template parameter (in this case our custom
unit system above, but ``parthenon::constants::SI`` and
``parthenon::constants::CGS`` are provided):

.. code:: c++

   parthenon::constants::PhysicalConstants<SIButKeVTemperatures> pc();
   std::cout << "Boltzmann constant: " << pc.kb << std::endl;

See also the unit test in `parthenon/tst/unit/test_unit_constants.cpp <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/unit/test_unit_constants.cpp>`_ for
more examples.
