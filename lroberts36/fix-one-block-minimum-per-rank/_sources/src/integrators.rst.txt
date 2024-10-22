.. _integrators:

Integrators
============

Integrators contain useful properties for writing a time-integration
scheme. They are defined in a class hierarchy. ``MultiStageDriver``
owns a pointer to an integrator. The type of integrator depends on the
template parameter. By default it is the base class, and you can
assign child integrator you like, but you must cast the pointer at
runtime to access useful features of the child classes. Alternatively
if you inherit from ``MultiStageDriver`` and specialize to a specific
child class, you can avoid doing the cast.

StagedIntegrator
------------------

The base class is the ``StagedIntegrator``. It contains a timestep
``dt``, a total number of integration steps, ``nstages``, a total
number of scratch buffers required, ``nbuffers``, and lists of strings
containing suggested names for the stages and buffers, ``buffer_name``
and ``stage_name``. All other integrators inherit from this one.

LowStorageIntegrator
----------------------


The ``LowStorageIntegrator`` contains integrators in the 2S form as
described in `Ketchson (2010)`_. These integrators are of the classic
`Shu Osher`_ form:

.. math::

   u^{(0)} &= u^n \\
   u^{(i)} &= \sum_{k=0}^{i-1} (\alpha_{i,k} u^{(k)} + \Delta t \beta_{i, k} F(u^{(k)})\\
   u^{n+1} &= u^{(m)}

where superscripts in parentheses mean subcycles in a Runge-Kutta
integration and :math:`F` is the right-hand-side of ODE system. The
difference between these low-storage methods and the classic SSPK
methods is that the low-storage methods typically have sparse
:math:`\alpha` and :math:`\beta` matrices, which are replaced by
diagonal termes, named :math:`\gamma_0` and :math:`\gamma_1`
respectively. 

These methods can be generalized to support more general methods with
the introduction of an additional :math:`\delta` term for first
averaging the updated stage with previous stages. This form is also described in Section 3.2.3 of the `Athena++ paper`_.

The full update then takes the form:

.. math::

   u^{(1)} &:= u^{(1)} + \delta_s u^{(0)} \\
   u^{(0)} &:= \gamma_{s0} u^{(0)} + \gamma_{s1} u^{(1)} + \beta_{s,s-1} \Delta t F(u^{(0)})

where here :math:`u^{(0)}` and :math:`u^{(1)}` are the two storage
buffers required to compute the update for a given Runge-Kutta stage
:math:`s`.

.. _Ketchson (2010): https://doi.org/10.1016/j.jcp.2009.11.006

.. _Shu Osher: https://doi.org/10.1016/0021-9991(88)90177-5

.. _Athena++ paper: https://doi.org/10.3847/1538-4365/ab929b

The ``LowStorageIntegrator`` contains arrays for ``delta``, ``beta``,
``gam0``, and ``gam1``. Available integration methods are:

* ``RK1``, which is simply forward Euler.

* ``RK2``, which is Heun's method.

* ``VL2``, 2nd-order Van Leer predictor-corrector from Stone and
  Gardiner 2009. Requires donor-cell reconstruction for the predictor
  stage.

* ``RK3``, a strong stability preserving variant.

* ``RK4``, a strong stability preserving variant.

ButcherIntegrator
---------------------

The ``ButcherIntegrator`` provides a classic Butcher Tableau with
arrays :math:`a` to compute the stages, :math:`c` to compute the time
offsets, and :math:`b` to compute the final update for a time
step. Available integration methods are:

* ``RK1``, simple forward Euler.

* ``RK2``, Heun's method.

* ``RK4``, The classic 4th-order method.

* ``RK10``, A recent version with fewer stages than Fehlberg's classic RK8(9), computed by Faegin and tabulated `here <https://sce.uhcl.edu/rungekutta/>`__.
