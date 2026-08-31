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

# This file was made in part with generative AI.

import numpy as np

import sys
import utils.test_case

sys.dont_write_bytecode = True


def sorted_positions(positions):
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    return positions[order]


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        if step in (1, 3):
            hook = "post_problem_generator" if step == 1 else "post_initialization"
            source_after_problem_generator = "true" if step == 1 else "false"
            parameters.driver_cmd_line_args = [
                f"parthenon/job/problem_id=particle_tracers_amr_{hook}_init",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
                "parthenon/time/tlim=0.0",
                f"Tracers/source_after_problem_generator={source_after_problem_generator}",
            ]
        elif step in (2, 4):
            hook = "post_problem_generator" if step == 2 else "post_initialization"
            source_after_problem_generator = "true" if step == 2 else "false"
            parameters.driver_cmd_line_args = [
                f"parthenon/job/problem_id=particle_tracers_amr_{hook}",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
                f"Tracers/source_after_problem_generator={source_after_problem_generator}",
            ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        from phdf import phdf

        initial_positions = {}
        for hook in ("post_problem_generator", "post_initialization"):
            initial = phdf(f"particle_tracers_amr_{hook}_init.out0.final.phdf")
            amr = phdf(f"particle_tracers_amr_{hook}.out0.final.phdf")

            initial_swarm = initial.GetSwarm("tracers")
            amr_swarm = amr.GetSwarm("tracers")
            initial_pos = np.vstack(
                (initial_swarm.x, initial_swarm.y, initial_swarm.z)
            ).transpose()
            amr_pos = np.vstack((amr_swarm.x, amr_swarm.y, amr_swarm.z)).transpose()

            # SourceTracers rounds each block's share independently, so the sum of
            # rounded allocations need not equal the requested global count. The 40-block
            # initialization mesh therefore contains 4104 particles for the requested 4096.
            expected_num_tracers = 4104
            if initial_pos.shape[0] != expected_num_tracers:
                print(f"Incorrect tracer count after {hook} initialization AMR.")
                print("expected:", expected_num_tracers, "actual:", initial_pos.shape[0])
                return False

            initial_positions[hook] = sorted_positions(initial_pos.copy())
            translated_initial_pos = initial_pos.copy()
            translated_initial_pos[:, 0] = (
                (translated_initial_pos[:, 0] + 0.5 + 0.35) % 1.0
            ) - 0.5
            translated_initial_pos = sorted_positions(translated_initial_pos)
            amr_pos = sorted_positions(amr_pos)

            if translated_initial_pos.shape != amr_pos.shape:
                print(f"Particle count changed during {hook} AMR tracer evolution.")
                print("initial:", translated_initial_pos.shape, "final:", amr_pos.shape)
                return False

            if not np.allclose(translated_initial_pos, amr_pos, atol=1.0e-10, rtol=0.0):
                diff = np.max(np.abs(translated_initial_pos - amr_pos))
                print(f"{hook} AMR tracer positions differ from the analytic translation.")
                print("max difference:", diff)
                return False

            initial_bounds = np.array(initial.BlockBounds)
            final_bounds = np.array(amr.BlockBounds)
            mesh_changed = initial.NumBlocks != amr.NumBlocks
            mesh_changed = mesh_changed or initial_bounds.shape != final_bounds.shape
            if not mesh_changed:
                mesh_changed = not np.allclose(initial_bounds, final_bounds)
            if not mesh_changed:
                print(f"AMR mesh did not change during the {hook} run.")
                return False

        if not np.allclose(
            initial_positions["post_problem_generator"],
            initial_positions["post_initialization"],
            atol=1.0e-10,
            rtol=0.0,
        ):
            print("Particle setup differs between PostProblemGenerator and PostInitialization.")
            return False

        return True
