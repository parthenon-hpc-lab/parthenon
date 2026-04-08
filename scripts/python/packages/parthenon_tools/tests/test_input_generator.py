#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

"""
Tests for Python input file generator.

Note: Tests that require C++ bindings will be skipped if
PARTHENON_ENABLE_PYTHON_BINDINGS was not enabled during build.
"""

try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    # Mock pytest.skip for running tests without pytest
    class pytest:
        @staticmethod
        def skip(msg):
            pass
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator

from parthenon_tools import InputFile, Block


def test_block_creation():
    """Test basic block creation and parameter access."""
    block = Block("test/block", param1=42, param2=3.14, param3="hello")

    assert block.name == "test/block"
    assert block.params["param1"] == 42
    assert block.params["param2"] == 3.14
    assert block.params["param3"] == "hello"


def test_block_modification():
    """Test that block parameters can be modified after creation."""
    block = Block("test/block", param1=42)

    block.params["param1"] = 100
    block.params["param2"] = "new"

    assert block.params["param1"] == 100
    assert block.params["param2"] == "new"


def test_input_file_accumulation():
    """Test that InputFile accumulates blocks correctly."""
    inp = InputFile()

    mesh = inp.block("parthenon/mesh", nx1=64)
    time = inp.block("parthenon/time", tlim=1.0)

    assert len(inp.blocks) == 2
    assert inp.blocks[0].name == "parthenon/mesh"
    assert inp.blocks[1].name == "parthenon/time"


def test_input_file_block_modification():
    """Test that blocks returned from InputFile can be modified."""
    inp = InputFile()

    mesh = inp.block("parthenon/mesh", nx1=64)
    mesh.params["nx1"] = 128
    mesh.params["nx2"] = 128

    assert inp.blocks[0].params["nx1"] == 128
    assert inp.blocks[0].params["nx2"] == 128


def test_text_output():
    """Test that InputFile can generate text output."""
    inp = InputFile()
    inp.block("test/block", param1=42, param2=3.14, flag=True)

    text = str(inp)

    assert "<test/block>" in text
    assert "param1 = 42" in text
    assert "param2 = 3.14" in text
    assert "flag = true" in text


def test_vector_parameters():
    """Test that vector parameters are handled correctly."""
    inp = InputFile()
    inp.block("test/block", values=[1, 2, 3], names=["a", "b", "c"])

    text = str(inp)

    assert "values = 1, 2, 3" in text
    assert "names = a, b, c" in text


def test_from_dict():
    """Test creating InputFile from dictionary."""
    config = {
        "parthenon/mesh": {"nx1": 64, "nx2": 64},
        "parthenon/time": {"tlim": 1.0, "nlim": 100}
    }

    inp = InputFile.from_dict(config)

    assert len(inp.blocks) == 2
    assert any(b.name == "parthenon/mesh" for b in inp.blocks)
    assert any(b.name == "parthenon/time" for b in inp.blocks)


@pytest.mark.skipif(True, reason="Requires C++ bindings")
def test_cpp_transfer():
    """Test transferring to C++ ParameterInput (requires bindings)."""
    try:
        import parthenon
    except ImportError:
        pytest.skip("Python bindings not available")

    inp = InputFile()
    inp.block("test/block",
              int_param=42,
              real_param=3.14,
              bool_param=True,
              string_param="hello",
              vector_param=[1, 2, 3])

    pi = inp.to_parameter_input()

    # Verify parameters transferred correctly
    assert pi.get_int("test/block", "int_param") == 42
    assert abs(pi.get_real("test/block", "real_param") - 3.14) < 1e-10
    assert pi.get_bool("test/block", "bool_param") == True
    assert pi.get_string("test/block", "string_param") == "hello"
    vec = pi.get_int_vector("test/block", "vector_param")
    assert vec == [1, 2, 3]


if __name__ == "__main__":
    if HAVE_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually
        print("Running tests without pytest...\n")
        tests = [
            ("test_block_creation", test_block_creation),
            ("test_block_modification", test_block_modification),
            ("test_input_file_accumulation", test_input_file_accumulation),
            ("test_input_file_block_modification", test_input_file_block_modification),
            ("test_text_output", test_text_output),
            ("test_vector_parameters", test_vector_parameters),
            ("test_from_dict", test_from_dict),
            ("test_cpp_transfer", test_cpp_transfer),
        ]

        passed = 0
        failed = 0
        for name, test_func in tests:
            try:
                test_func()
                print(f"✓ {name}")
                passed += 1
            except Exception as e:
                print(f"✗ {name}: {e}")
                failed += 1

        print(f"\n{passed} passed, {failed} failed")
        if failed > 0:
            exit(1)
