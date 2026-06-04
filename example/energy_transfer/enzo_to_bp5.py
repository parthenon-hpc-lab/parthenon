#!/usr/bin/env python3
"""
Convert Enzo simulation data to ADIOS2/bp5 format for the Parthenon
energy transfer analysis application.

Reads fields via yt (covering_grid) and writes them in k,j,i (C-order)
layout matching Parthenon's UniformGridHelper::FlatIndex convention.

Usage:
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --gamma 1.001
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --res 64 --gamma 1.001
"""

import argparse
import numpy as np
import adios2


def main():
    parser = argparse.ArgumentParser(
        description="Convert Enzo data to ADIOS2/bp5 for Parthenon energy transfer"
    )
    parser.add_argument("data_path", help="Path to Enzo data (e.g., DD0024/data0024)")
    parser.add_argument("--output", default="enzo_data.bp", help="Output bp5 filename")
    parser.add_argument(
        "--res",
        type=int,
        default=None,
        help="Target resolution (default: native resolution)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Adiabatic index for pressure computation (if not set, no pressure output)",
    )
    parser.add_argument(
        "--no-acc",
        action="store_true",
        help="Skip acceleration fields",
    )
    args = parser.parse_args()

    import yt

    yt.set_log_level("warning")

    print(f"Loading {args.data_path} ...")
    ds = yt.load(args.data_path)

    native_res = int(ds.domain_dimensions[0])
    res = args.res if args.res is not None else native_res
    print(f"Native resolution: {native_res}^3, output resolution: {res}^3")

    domain_left = ds.domain_left_edge.d
    domain_right = ds.domain_right_edge.d
    Lx = domain_right[0] - domain_left[0]

    all_data = ds.covering_grid(
        level=0, left_edge=domain_left, dims=[native_res] * 3
    )

    def get_field(field_name):
        # yt returns [x, y, z] order (Fortran); transpose to [z, y, x] = [k, j, i]
        data = np.float64(all_data[field_name].d.T)
        if res != native_res:
            data = downsample(data, native_res, res)
        return np.ascontiguousarray(data)

    def downsample(arr, from_res, to_res):
        factor = from_res // to_res
        if factor == 1:
            return arr
        # Volume averaging
        result = arr.reshape(to_res, factor, to_res, factor, to_res, factor)
        return result.mean(axis=(1, 3, 5))

    print("Reading density...")
    rho = get_field("Density")

    print("Reading velocity...")
    vx = get_field("x-velocity")
    vy = get_field("y-velocity")
    vz = get_field("z-velocity")

    print("Reading magnetic field...")
    bx = get_field("Bx")
    by = get_field("By")
    bz = get_field("Bz")

    has_acc = not args.no_acc
    if has_acc:
        try:
            print("Reading acceleration...")
            ax = get_field("x-acceleration")
            ay = get_field("y-acceleration")
            az = get_field("z-acceleration")
        except Exception as e:
            print(f"  Warning: acceleration not available ({e}), skipping")
            has_acc = False

    has_pres = args.gamma is not None
    if has_pres:
        print("Reading pressure (derived from internal energy)...")
        pres = get_field("pressure")

    # Write ADIOS2/bp5 file
    # All 3D fields stored as [Nz, Ny, Nx] shape (k,j,i order)
    print(f"Writing {args.output} ...")
    with adios2.Stream(args.output, "w") as stream:
        stream.begin_step()

        # Metadata attributes
        stream.write_attribute("resolution", np.array([res, res, res], dtype=np.int32))
        stream.write_attribute(
            "domain_left", np.array(domain_left, dtype=np.float64)
        )
        stream.write_attribute(
            "domain_right", np.array(domain_right, dtype=np.float64)
        )
        if args.gamma is not None:
            stream.write_attribute("gamma", args.gamma)

        # Scalar fields
        stream.write("rho", rho, rho.shape, [0, 0, 0], rho.shape)

        # Velocity components
        stream.write("vel_x", vx, vx.shape, [0, 0, 0], vx.shape)
        stream.write("vel_y", vy, vy.shape, [0, 0, 0], vy.shape)
        stream.write("vel_z", vz, vz.shape, [0, 0, 0], vz.shape)

        # Magnetic field components
        stream.write("mag_x", bx, bx.shape, [0, 0, 0], bx.shape)
        stream.write("mag_y", by, by.shape, [0, 0, 0], by.shape)
        stream.write("mag_z", bz, bz.shape, [0, 0, 0], bz.shape)

        # Acceleration (optional)
        if has_acc:
            stream.write("acc_x", ax, ax.shape, [0, 0, 0], ax.shape)
            stream.write("acc_y", ay, ay.shape, [0, 0, 0], ay.shape)
            stream.write("acc_z", az, az.shape, [0, 0, 0], az.shape)

        # Pressure (optional)
        if has_pres:
            stream.write("pres", pres, pres.shape, [0, 0, 0], pres.shape)

        stream.end_step()

    print("Done.")
    print(f"  Fields written: rho, vel_{{x,y,z}}, mag_{{x,y,z}}", end="")
    if has_acc:
        print(", acc_{x,y,z}", end="")
    if has_pres:
        print(", pres", end="")
    print()
    print(f"  Shape: ({res}, {res}, {res}) [k, j, i order]")
    print(f"  Domain: [{domain_left[0]}, {domain_right[0]}]^3, L = {Lx}")


if __name__ == "__main__":
    main()
