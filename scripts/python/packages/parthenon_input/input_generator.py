#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
# This file was made in part with generative AI.

"""
Python-based input file generator for Parthenon.

Provides a two-stage approach:
1. Build mutable parameter structure in Python
2. Transfer to typed C++ ParameterInput when ready

Example usage:
    from parthenon_input import InputFile

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
        self.params = {}
        self._typed = {}  # Track which params are typed (True) vs from file (False)

        for key, value in params.items():
            self.params[key] = value
            self._typed[key] = True  # Parameters passed at construction are typed

    def set(self, **params: Any) -> None:
        """
        Set or update parameters with strong typing.

        Args:
            **params: Parameter key-value pairs to set

        Example:
            block.set(nx1=128, nx2=128, x1min=0.0)
        """
        for key, value in params.items():
            self.params[key] = value
            self._typed[key] = True  # Explicitly set parameters are typed

    def _set_from_file(self, key: str, value: str) -> None:
        """Internal: Set parameter from file (unresolved string)."""
        self.params[key] = value
        self._typed[key] = False  # From file, will need lazy conversion

    def is_typed(self, key: str) -> bool:
        """Check if a parameter is strongly typed (vs loaded from file)."""
        return self._typed.get(key, True)

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
            mesh.set(nx2=128)  # Can modify after creation
        """
        blk = Block(name, **params)
        self.blocks.append(blk)
        return blk

    def get_block(self, name: str) -> Optional[Block]:
        """
        Get a block by name.

        Args:
            name: Block name (e.g., "parthenon/mesh")

        Returns:
            Block object if found, None otherwise

        Example:
            mesh = inp.get_block("parthenon/mesh")
            if mesh:
                mesh.set(nx1=128)
        """
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def to_parameter_input(self, pi=None):
        """
        Transfer to C++ ParameterInput with full type preservation.

        This dispatches each parameter to the appropriate AddParsedParameter() method
        based on its Python type:
            - int -> add_int()
            - float -> add_real()
            - bool -> add_bool()
            - str -> add_string()
            - list[int] -> add_int_vector()
            - etc.

        Args:
            pi: Optional ParameterInput object to populate. If None, creates a new one.

        Returns:
            Pybind11-wrapped ParameterInput object (either provided or newly created)

        Example:
            inp = InputFile()
            inp.block("parthenon/mesh", nx1=64, x1min=0.0)
            pi = inp.to_parameter_input()

        Example with existing ParameterInput:
            # pi provided by C++ code
            inp = InputFile()
            inp.block("parthenon/mesh", nx1=64)
            inp.to_parameter_input(pi)  # populate existing pi
        """
        try:
            import parthenon
        except ImportError:
            raise ImportError(
                "parthenon module not found. "
                "Make sure pybind11 bindings are built and installed."
            )

        if pi is None:
            pi = parthenon.ParameterInput()

        for block in self.blocks:
            for key, value in block.params.items():
                is_typed = block.is_typed(key)
                self._add_typed_parameter(pi, block.name, key, value, is_typed)

        # NOTE: Don't call finalize_parsing() here - application may still want to
        # call ModifyFromCmdline() or other parsing. Application should call
        # finalize_parsing() or let first Get/GetOrAdd call it automatically.

        return pi

    def _add_typed_parameter(
        self, pi, block_name: str, param_name: str, value: Any, is_typed: bool = True
    ):
        """
        Dispatch to appropriate AddParsedParameter method based on parameter type and origin.

        Args:
            pi: C++ ParameterInput object
            block_name: Block name
            param_name: Parameter name
            value: Parameter value
            is_typed: If False, value is from file and should use UnresolvedString
        """
        # If parameter came from file, use unresolved string for lazy conversion
        if not is_typed:
            pi.add_unresolved(block_name, param_name, str(value))
            return

        # Otherwise dispatch based on Python type
        if isinstance(value, bool):
            # Must check bool before int (bool is subclass of int in Python)
            pi.add_bool(block_name, param_name, value)
        elif isinstance(value, int):
            pi.add_int(block_name, param_name, value)
        elif isinstance(value, float):
            pi.add_real(block_name, param_name, value)
        elif isinstance(value, str):
            pi.add_string(block_name, param_name, value)
        elif isinstance(value, (list, tuple)):
            # Dispatch vector based on element type
            if len(value) == 0:
                raise ValueError(
                    f"Cannot infer type of empty list for {block_name}/{param_name}"
                )
            first = value[0]
            if isinstance(first, bool):
                pi.add_bool_vector(block_name, param_name, list(value))
            elif isinstance(first, int):
                pi.add_int_vector(block_name, param_name, list(value))
            elif isinstance(first, float):
                pi.add_real_vector(block_name, param_name, list(value))
            elif isinstance(first, str):
                pi.add_string_vector(block_name, param_name, list(value))
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
    def from_dict(
        config: Dict[str, Dict[str, Any]], header: Optional[str] = None
    ) -> "InputFile":
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

    @staticmethod
    def from_file(filename: str) -> "InputFile":
        """
        Load an existing Parthenon input file.

        Parameters loaded from file are stored as unresolved strings
        for lazy type conversion (matching C++ behavior). Parameters
        subsequently modified in Python become strongly typed.

        Args:
            filename: Path to .pin file

        Returns:
            InputFile object with parameters loaded from file

        Example:
            inp = InputFile.from_file("base.pin")
            mesh = inp.get_block("parthenon/mesh")
            mesh.set(nx1=128)  # This override becomes strongly typed
            pi = inp.to_parameter_input()
        """
        inp = InputFile()
        current_block = None

        with open(filename, "r") as f:
            for line in f:
                # Strip whitespace and comments
                line = line.split("#")[0].strip()
                if not line:
                    continue

                # Check for block header
                if line.startswith("<") and line.endswith(">"):
                    block_name = line[1:-1]
                    current_block = Block(block_name)
                    inp.blocks.append(current_block)
                    continue

                # Parse parameter line
                if "=" in line and current_block is not None:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Store as unresolved string (lazy conversion)
                    current_block._set_from_file(key, value)

        return inp
