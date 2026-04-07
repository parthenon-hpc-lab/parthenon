# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import numpy as np

import sys
import utils.test_case

sys.dont_write_bytecode = True


def sorted_positions(positions):
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    return positions[order]


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        if step == 1:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=particle_tracers_amr_init",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
                "parthenon/time/tlim=0.0",
            ]
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=particle_tracers_amr",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
            ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        from phdf import phdf

        initial = phdf("particle_tracers_amr_init.out0.final.phdf")
        amr = phdf("particle_tracers_amr.out0.final.phdf")

        amr_swarm = amr.GetSwarm("tracers")
        initial_swarm = initial.GetSwarm("tracers")

        initial_pos = np.vstack(
            (initial_swarm.x, initial_swarm.y, initial_swarm.z)
        ).transpose()
        amr_pos = np.vstack((amr_swarm.x, amr_swarm.y, amr_swarm.z)).transpose()

        initial_pos[:, 0] = ((initial_pos[:, 0] + 0.5 + 0.35) % 1.0) - 0.5
        initial_pos = sorted_positions(initial_pos)
        amr_pos = sorted_positions(amr_pos)

        if initial_pos.shape != amr_pos.shape:
            print("Particle count changed during AMR tracer evolution.")
            print("initial:", initial_pos.shape, "final:", amr_pos.shape)
            return False

        if not np.allclose(initial_pos, amr_pos, atol=1.0e-10, rtol=0.0):
            diff = np.max(np.abs(initial_pos - amr_pos))
            print("AMR tracer positions differ from the analytic translation.")
            print("max difference:", diff)
            return False

        initial_bounds = np.array(initial.BlockBounds)
        final_bounds = np.array(amr.BlockBounds)
        mesh_changed = initial.NumBlocks != amr.NumBlocks
        mesh_changed = mesh_changed or initial_bounds.shape != final_bounds.shape
        if not mesh_changed:
            mesh_changed = not np.allclose(initial_bounds, final_bounds)
        if not mesh_changed:
            print("AMR mesh did not change between initialization and final output.")
            return False

        return True
