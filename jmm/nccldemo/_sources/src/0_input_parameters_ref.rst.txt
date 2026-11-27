.. _inputref:

Input Parameters Reference
===========================

This reference is automatically generated with Parthenon's
``ParameterInput`` class. To generate it, pass the ``-p`` flag into a
parthenon-based executable (in addition to the other flags you would
normally pass), and optionally a ``regex`` to specify which blocks
you'd like to output. Parthenon will print a valid CSV file to the
terminal.

Every call to ``ParameterInput::Get*`` optionally takes a "docstring"
as a final argument, which is the details column listed here. Default
values are recorded if they are available.

In Parthenon
--------------

An incomplete list of all input parameters provided by Parthenon is tabulated below

.. csv-table:: Parthenon input parameters
   :file: generated/diffusion-parth-table.csv
   :header-rows: 1
   :widths: 20 20 10 20 30
   :class: csv-wrap

In the diffusion example
----------------------------

The non-parthenon input parameters used in the diffusion example are tabulated below

.. csv-table:: Parthenon input parameters
   :file: generated/diffusion-table.csv
   :header-rows: 1
   :widths: 20 20 10 20 30
   :class: csv-wrap

In the Burgers benchmark, Parthenon-VIBE
------------------------------------------

The non-parthenon input parameters used in Parthenon-VIBE are tabulated below

.. csv-table:: Parthenon input parameters
   :file: generated/burgers-table.csv
   :header-rows: 1
   :widths: 20 20 10 20 30
   :class: csv-wrap
