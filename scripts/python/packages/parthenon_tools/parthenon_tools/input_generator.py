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
Python-based input file generator for Parthenon.

Provides a two-stage approach:
1. Build mutable parameter structure in Python
2. Transfer to typed C++ ParameterInput when ready

Example usage:
    from parthenon_tools.input_generator import InputFile

    inp = InputFile()
    mesh = inp.block("parthenon/mesh", nx1=64, nx2=64, nx3=64)
    mesh.params["x1min"] = 0.0  # Can modify after creation

    inp.block("parthenon/time", tlim=1.0, nlim=100)

    # Transfer to C++ ParameterInput with type preservation
    pi = inp.to_parameter_input()
"""

from typing import Any, Dict, List, Optional


class Block:
    """Represents a parameter block in a Parthenon input file."""

    def __init__(self, name: str, **params: Any):
        """
        Create a parameter block.

        Args:
            name: Block name (e.g., "parthenon/mesh")
            **params: Parameter key-value pairs
        """
        self.name = name
        self.params = params

    def _format_value(self, value: Any) -> str:
        """Convert a Python value to input file format."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            # Handle vectors
            return ", ".join(str(v) for v in value)
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    def to_string(self) -> str:
        """Convert block to input file format."""
        lines = [f"<{self.name}>"]
        for key, value in self.params.items():
            formatted = self._format_value(value)
            lines.append(f"{key} = {formatted}")
        return "\n".join(lines)


class InputFile:
    """
    Accumulator for parameter blocks that can transfer to C++ ParameterInput.

    This provides a two-stage approach:
    1. Build mutable parameter structure in Python
    2. Transfer to typed C++ ParameterInput when ready

    Example:
        inp = InputFile()
        mesh = inp.block("parthenon/mesh", nx1=64, nx2=64)
        mesh.params["x1min"] = 0.0  # Can modify after creation

        # Transfer to C++ with type preservation
        pi = inp.to_parameter_input()
    """

    def __init__(self, header: Optional[str] = None):
        """
        Create an input file builder.

        Args:
            header: Optional comment header to include at top of file
        """
        self.blocks: List[Block] = []
        self.header = header

    def block(self, name: str, **params: Any) -> Block:
        """
        Add a parameter block and return it for further modification.

        Args:
            name: Block name (e.g., "parthenon/mesh")
            **params: Parameter key-value pairs

        Returns:
            Block object (already added to this InputFile)

        Example:
            inp = InputFile()
            mesh = inp.block("parthenon/mesh", nx1=64)
            mesh.params["nx2"] = 128  # Can modify after creation
        """
        blk = Block(name, **params)
        self.blocks.append(blk)
        return blk

    def to_parameter_input(self):
        """
        Transfer to C++ ParameterInput with full type preservation.

        This dispatches each parameter to the appropriate Set<T>() method
        based on its Python type:
            - int -> Set<int>()
            - float -> Set<Real>()
            - bool -> Set<bool>()
            - str -> Set<std::string>()
            - list[int] -> Set<std::vector<int>>()
            - etc.

        Returns:
            Pybind11-wrapped ParameterInput object

        Example:
            inp = InputFile()
            inp.block("parthenon/mesh", nx1=64, x1min=0.0)
            pi = inp.to_parameter_input()
        """
        try:
            import parthenon
        except ImportError:
            raise ImportError(
                "parthenon module not found. "
                "Make sure pybind11 bindings are built and installed."
            )

        pi = parthenon.ParameterInput()

        for block in self.blocks:
            for key, value in block.params.items():
                self._set_typed_parameter(pi, block.name, key, value)

        return pi

    def _set_typed_parameter(self, pi, block_name: str, param_name: str, value: Any):
        """Dispatch to appropriate Set<T>() method based on Python type."""
        if isinstance(value, bool):
            # Must check bool before int (bool is subclass of int in Python)
            pi.set_bool(block_name, param_name, value)
        elif isinstance(value, int):
            pi.set_int(block_name, param_name, value)
        elif isinstance(value, float):
            pi.set_real(block_name, param_name, value)
        elif isinstance(value, str):
            pi.set_string(block_name, param_name, value)
        elif isinstance(value, (list, tuple)):
            # Dispatch vector based on element type
            if len(value) == 0:
                raise ValueError(f"Cannot infer type of empty list for {block_name}/{param_name}")
            first = value[0]
            if isinstance(first, bool):
                pi.set_bool_vector(block_name, param_name, list(value))
            elif isinstance(first, int):
                pi.set_int_vector(block_name, param_name, list(value))
            elif isinstance(first, float):
                pi.set_real_vector(block_name, param_name, list(value))
            elif isinstance(first, str):
                pi.set_string_vector(block_name, param_name, list(value))
            else:
                raise TypeError(f"Unsupported vector element type: {type(first)}")
        else:
            raise TypeError(f"Unsupported parameter type: {type(value)}")

    def __str__(self) -> str:
        """
        Generate text representation (for debugging or fallback).

        Note: For production use, prefer to_parameter_input() which preserves types.
        """
        lines = []

        if self.header:
            for line in self.header.split("\n"):
                lines.append(f"# {line}")
            lines.append("")

        for i, blk in enumerate(self.blocks):
            lines.append(blk.to_string())
            # Add blank line between blocks (but not after last one)
            if i < len(self.blocks) - 1:
                lines.append("")

        return "\n".join(lines) + "\n"

    def write(self, filename: str) -> None:
        """
        Write text representation to disk (for debugging or fallback).

        Note: For production use, prefer to_parameter_input() which preserves types.

        Args:
            filename: Output filename
        """
        with open(filename, "w") as f:
            f.write(str(self))

    @staticmethod
    def from_dict(config: Dict[str, Dict[str, Any]], header: Optional[str] = None) -> "InputFile":
        """
        Create an InputFile from a nested dictionary.

        Useful for loading from JSON/YAML.

        Args:
            config: Dictionary of block_name -> {param: value}
            header: Optional comment header

        Returns:
            InputFile object

        Example:
            config = {
                "parthenon/mesh": {"nx1": 64, "nx2": 64},
                "parthenon/time": {"tlim": 1.0}
            }
            inp = InputFile.from_dict(config)
            pi = inp.to_parameter_input()
        """
        inp = InputFile(header=header)
        for name, params in config.items():
            inp.block(name, **params)
        return inp
