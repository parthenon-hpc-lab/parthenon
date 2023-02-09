.. Spiner Documentation master file, created by
   sphinx-quickstart on Tue Nov 2 16:56:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Parthenon: A Performance Portable Block-Structured Adaptive Mesh Refinement Framework
=====================================================================================

`Parthenon`_ is a performance portable block-structured adaptive mesh refinement framework.

.. _Parthenon: https://github.com/parthenon-hpc-lab/parthenon

We are currently in the process of migrating to new documentation
here. The old documentation can be found in markdown `here`_.

.. _`here`: https://github.com/parthenon-hpc-lab/parthenon/docs

Key Features
^^^^^^^^^^^^^

* Device first/device resident approach (work data only in device memory to prevent expensive transfers between host and device)
* Transparent packing of data across blocks (to reduce/hide kernel launch latency)
* Direct device-to-device communication via asynchronous, one-sided  MPI communication
* Intermediate abstraction layer to hide complexity of device kernel launches
* Flexible, plug-in package system
* Abstract variables controlled via metadata flags
* Support for particles
* Multi-stage drivers/integrators with support for task-based parallelism

Community
^^^^^^^^^^

* `Chat room on matrix.org`_

.. _`Chat room on matrix.org`: https://app.element.io/#/room/#parthenon-general:matrix.org


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   src/metadata
   src/sphinx-doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
