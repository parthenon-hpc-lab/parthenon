.. _rummy_input:

Rummy Input Files
=================

Parthenon supports an extended input file format provided by the `Link Rummy <https://github.com/lanl/rummy>` library, in addition
to the native, Athena++ format. Rummy input files provide expression evaluation,
global variables, vector operations, relative block paths, file inclusion,
and more.

Rummy is auto-detected — no special build flag is required.  Whether a
particular input file is parsed by Rummy or the native parser is determined
automatically at runtime (see :ref:`rummy_detection`).

.. contents::
   :local:
   :depth: 2


Basic Syntax
------------

Rummy input files share the same block/parameter structure as native files:

.. code-block:: text

   <blockname>
   param = value           # optional inline comment
   other  = 1.23

The key difference is that Rummy compiles the entire file from the top down,
so any card defined earlier can be referenced by name in a later expression.


Global Variables
----------------

Variables declared **before** the first ``<block>`` header are *global variables*.
Global variables can be referenced by name anywhere in
the file without a block qualifier:

.. code-block:: text

   L   = 1.0
   rho = 2.5

   <mesh>
   Lx = L          # reference the global variable L
   Ly = L

Global variables are the simplest Rummy feature.  Their presence (content
before the first ``<...>`` line) is one of the markers that causes Parthenon
to choose the Rummy parser automatically.


Expression Evaluation
---------------------

Parameter values can be arbitrary arithmetic expressions.

**Arithmetic**

+------------+-------------------------------+
| Syntax     | Meaning                       |
+============+===============================+
| ``+  -``   | Addition / subtraction        |
+------------+-------------------------------+
| ``*  /``   | Multiplication / division     |
+------------+-------------------------------+
| ``//``     | Integer (floor) division      |
+------------+-------------------------------+
| ``**``     | Power (e.g. ``2**10``)        |
+------------+-------------------------------+
| ``%``      | Modulo                        |
+------------+-------------------------------+
| ``pi``     | Named constant π              |
+------------+-------------------------------+

**Boolean** (operate on ``true``/``false`` values)

+----------+----------------------+
| Syntax   | Meaning              |
+==========+======================+
| ``and``  | Logical AND          |
+----------+----------------------+
| ``or``   | Logical OR           |
+----------+----------------------+
| ``xor``  | Logical XOR          |
+----------+----------------------+
| ``not``  | Logical NOT (unary)  |
+----------+----------------------+

**Bitwise** (operate on integers; also work on booleans)

+--------+---------------------+
| Syntax | Meaning             |
+========+=====================+
| ``&``  | Bitwise AND         |
+--------+---------------------+
| ``|``  | Bitwise OR          |
+--------+---------------------+
| ``^``  | Bitwise XOR         |
+--------+---------------------+
| ``~``  | Bitwise NOT (unary) |
+--------+---------------------+
| ``<<`` | Left shift          |
+--------+---------------------+
| ``>>`` | Right shift         |
+--------+---------------------+

**Comparison**: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``

Examples:

.. code-block:: text

   dt      = 0.45 * dx
   gamma   = 5.0/3.0
   cv      = 1.0/(gamma - 1.0)
   vol     = L**3
   half_nx = nx // 2
   use_mhd = hydro and conduction

Values from **any previously-defined card** (including cards in other blocks)
can be referenced by their fully-qualified dotted name:

.. code-block:: text

   <eos>
   gamma = 5.0/3.0

   <gas>
   cv = 1.0/(eos.gamma - 1.0)


Math Functions
--------------

Almost all ``<cmath>`` functions are available as built-in keywords:

**Trigonometric** — ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``,
``atan2(y, x)``

**Exponential / logarithm** — ``exp``, ``log`` (natural), ``log10``

**Power / rounding** — ``sqrt``, ``ceil``, ``floor``, ``abs``, ``sign``

**Extrema** — ``min(a, b)``, ``max(a, b)``

.. code-block:: text

   theta = pi / 4.0
   vx    = v * cos(theta)
   vy    = v * sin(theta)
   r     = sqrt(vx**2 + vy**2)
   lo    = min(r, 1.0)


Environment Variables
---------------------

The special function ``env("VARIABLE")`` is available for reading environment variables. The output can be stored into a variable for later use.

Ternary Operator
----------------

The C-style ternary ``condition ? value_if_true : value_if_false`` is
supported:

.. code-block:: text

   nx = 64
   ny = (nx > 32) ? nx // 2 : nx       # ny = 32
   label = (debug) ? "debug" : "production"


String Parameters
-----------------

String values must be quoted:

.. code-block:: text

   <parthenon/job>
   problem_id = "advection"

String concatenation uses ``+``:

.. code-block:: text

   prefix = "my_"
   <outputs>
   label = prefix + "run"


Boolean Parameters
------------------

Boolean values are written as ``true`` or ``false`` (case-insensitive):

.. code-block:: text

   <physics>
   hydro      = true
   conduction = false
   do_work    = hydro or conduction
   both       = hydro and conduction
   neither    = not hydro and not conduction


Vector Parameters
-----------------

Vectors are comma-separated lists.  Both bare and bracketed syntax work:

.. code-block:: text

   L = 1.0, 1.0, 0.5       # bare comma list
   n = [10, 10, 1]          # bracket syntax

Individual elements are accessed with zero-based indexing:

.. code-block:: text

   <mesh>
   nx1 = n[0]
   nx2 = n[1]

**Slice assignments** copy a range of elements from a vector into another:

.. code-block:: text

   xmin = -L[0]/2., -L[1]/2., -L[2]/2.
   xmax[:] = 0.5 * L[:3]     # broadcast scalar * slice

The ``[:]`` slice on the left-hand side means "all elements"; ``[:3]`` means
elements 0, 1, 2.


Relative Block Paths
--------------------

A block header starting with ``<../`` declares a child of the *current* block:

.. code-block:: text

   <gas>
   name = "hydrogen"

   <../eos>            # expands to <gas/eos>
   gamma = 5.0/3.0

   <../conductivity>   # expands to <gas/conductivity>
   kappa = 0.1 / gas.eos.gamma

Relative paths allow logically related sub-blocks to stay near each other in
the file without repeating long prefixes.


Including Other Files
---------------------

The ``include`` statement inserts another file into the current compilation
at that point.  All variables defined before the ``include`` are visible
inside the included file, and all variables defined inside are visible after
it returns:

.. code-block:: text

   # main.par
   # use rummy

   L  = 1.0
   nx = 64

   include "mesh.par"            # relative to the directory of main.par
   include "/abs/path/eos.par"   # absolute path also works

Circular includes are detected and cause a fatal error.  Paths are resolved
relative to the directory of the file containing the ``include`` statement.


Multiple Input Files
--------------------

Multiple ``-i`` arguments can be passed on the command line.  All files are
compiled in the **same Rummy compilation space**, so variables defined in an
earlier file are available in later ones:

.. code-block:: bash

   ./my-app -i base.par -i overrides.par parthenon.time.nlim=100

.. code-block:: text

   # base.par
   # use rummy
   nx = 64
   <parthenon/mesh>
   nx1 = nx

.. code-block:: text

   # overrides.par — sees nx from base.par
   nx = 128         # redefines nx for the higher-resolution run

Files are read in the order they appear on the command line.


Debugging Utilities
-------------------

Three special statements print the current state of the compiler to
standard output and are useful for debugging input decks:

``__locals__``
   Print all variables local to the current block (suit).

``__globals__``
   Print all globally defined variables (including cards from all
   previously compiled suits).

``__stack__``
   Print the current expression evaluation stack.

``__list__``
   The combined output of ``__globals__``, ``__locals__``, and ``__stack__``.

.. code-block:: text

   <mesh>
   nx = 64
   __globals__     # prints all globals including nx


The ``print`` Function
----------------------

The built-in variadic ``print`` function writes any previously compiled card's value to
standard output when the deck is loaded.  It is useful for sanity-checking
derived quantities:

.. code-block:: text

   dt = 0.45 * dx
   print("dt = ", dt)

Only cards defined *before* the ``print`` call can be printed.


Multiline Expressions
---------------------

A trailing ``&`` continues an expression on the next line:

.. code-block:: text

   long_value = 1.0   &
              + 2.0   &
              + 3.0   # = 6.0


.. _rummy_detection:

How Parthenon Detects Rummy Files
----------------------------------

Parthenon auto-detects Rummy format without any explicit flag.  A file (or
command-line override string) is routed to the Rummy parser if **any** of the
following is true:

1. The first non-blank line is ``# use rummy`` (case-insensitive) — the
   explicit opt-in marker.
2. There is non-comment, non-blank content **before** the first ``<block>``
   header (i.e., global variables are present).
3. A block header begins with ``<..`` (relative path syntax).
4. A parameter **name** contains ``.`` or ``[`` (dotted reference or vector
   index on the LHS).
5. A parameter **value** contains any of: ``**``, ``"``, ``[``, ``%``, ``^``, or ``|`` (expression operators or quoted strings).

To unconditionally use the native (Rummy) parser, place ``# use native`` (``# use rummy``) at the first line of the file:

.. code-block:: text

   # use rummy
   <parthenon/time>
   nlim = 100
   tlim = 1.0

A native input file (no Rummy syntax) continues to be parsed by the native
parser transparently.


Restart Files and Rummy
-----------------------

When restarting from an HDF5 restart file (``-r``), Parthenon loads the
parameter snapshot stored in the restart file using the native parser, then
layers any Rummy input file (``-i``) and command-line overrides on top.

The Rummy deck is seeded with all parameters from the restart before the Rummy
file is compiled, so expressions in the Rummy file can reference values that
came from the restart.  For example:

.. code-block:: bash

   ./my-app -r run.out1.final.rhdf -i params.par parthenon.time.nlim=10

``parthenon.time.nlim=10`` is itself detected as a Rummy override (native
overrides use ``block/param=value`` syntax) and is applied after the restart
parameters are seeded into the deck.


Example
-------

A complete minimal Rummy input file:

.. code-block:: text

   # use rummy

   # --- Global parameters ---
   L  = 1.0
   nx = 64

   include "common_physics.par"

   <parthenon/mesh>
   nx1   = nx
   nx2   = nx
   x1min = -L/2.
   x1max =  L/2.
   x2min = -L/2.
   x2max =  L/2.

   <parthenon/time>
   tlim = 2.0
   nlim = -1
   dt   = 0.45 * L / nx    # CFL-based initial guess

   <../output>              # relative: expands to parthenon/output
   file_type = "hdf5"
   dt        = 0.1

   __globals__              # print all compiled globals for debugging
