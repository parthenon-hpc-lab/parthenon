.. _signals:

Signals
=======

``Parthenon`` handles several operating system signals.  Generally, it attempts
to exit gracefully (saving simulation state), but upon receiving repeated
signals it will exit immediately like any other program.

-  If ``Parthenon`` catches a ``SIGINT``, ``SIGTERM``, or ``SIGALRM``, it will
   attempt to exit gracefully at the end of the current step, after
   outputting any ``final`` simulation output files.  These are the signals
   sent with ``^C``, by Slurm when a job has reached its walltime limit,
   or after the time specified on the command line with ``-t HH:MM:SS``.
-  If ``Parthenon`` catches three ``SIGINT`` or ``SIGTERM`` signals during a
   single step, it will exit immediately with a ``PARTHENON_ERROR``.  So, if
   an application is misbehaving, **three** ``SIGINT``s with ``^C`` will end
   it immediately.

In addition to signals, ``Parthenon`` applications will react to special files
placed in the output directory.

-  Creating/touching a new file with the name ``output_now`` in
   the working directory of a running application will cause it to
   write output files with a ``now`` id rather than a number. After the
   output is written. the ``output_now`` file is removed and the
   simulation continues normally. The user can repeat the process any
   time by creating a new ``output_now`` file.
-  Creating a file named ``stop_now`` will stop a running simulation
   at the end of the current step, in exactly the same way as sending
   a single ``SIGINT``.

Signal handling is implemented in ``utils/signal_handler.cpp``.