#!/usr/bin/env python3
"""
Convert Enzo simulation data to ADIOS2/bp5 format for the Parthenon
energy transfer analysis application.

Reads fields via yt (covering_grid) and writes them in k,j,i (C-order)
layout matching Parthenon's UniformGridHelper::FlatIndex convention.

Usage:
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --res 64
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --precision single
    python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --quantity-type conserved
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
        "--no-acc",
        action="store_true",
        help="Skip acceleration fields",
    )
    parser.add_argument(
        "--no-pres",
        action="store_true",
        help="Skip pressure field in primitive output",
    )
    parser.add_argument(
        "--precision",
        choices=("double", "single"),
        default="double",
        help="Floating-point precision for field data written to ADIOS2 (default: double)",
    )
    parser.add_argument(
        "--quantity-type",
        choices=("primitive", "conserved"),
        default="primitive",
        help=(
            "Write primitive fields rho, vel_{x,y,z}, mag_{x,y,z}, pres or conserved "
            "fields rho, mom_{x,y,z}, mag_{x,y,z}, total_energy (default: primitive)"
        ),
    )
    parser.add_argument("--rho-field", default="Density", help="yt density field")
    parser.add_argument(
        "--vel-fields",
        nargs=3,
        default=["x-velocity", "y-velocity", "z-velocity"],
        metavar=("VX", "VY", "VZ"),
        help="yt velocity fields for primitive output",
    )
    parser.add_argument(
        "--mom-fields",
        nargs=3,
        default=[
            "gas:momentum_density_x",
            "gas:momentum_density_y",
            "gas:momentum_density_z",
        ],
        metavar=("MX", "MY", "MZ"),
        help="yt momentum-density fields for conserved output",
    )
    parser.add_argument(
        "--mag-fields",
        nargs=3,
        default=["Bx", "By", "Bz"],
        metavar=("BX", "BY", "BZ"),
        help="yt magnetic-field fields",
    )
    parser.add_argument(
        "--total-energy-field",
        default="gas:specific_total_energy",
        help=(
            "yt specific total energy field for conserved output; it is multiplied by "
            "density and written as total_energy"
        ),
    )
    parser.add_argument(
        "--pressure-field",
        default="pressure",
        help="yt pressure field for primitive output",
    )
    parser.add_argument(
        "--acc-fields",
        nargs=3,
        default=["x-acceleration", "y-acceleration", "z-acceleration"],
        metavar=("AX", "AY", "AZ"),
        help="yt acceleration fields",
    )
    args = parser.parse_args()
    output_dtype = np.float32 if args.precision == "single" else np.float64

    import yt

    yt.set_log_level("warning")

    print(f"Loading {args.data_path} ...")
    ds = yt.load(args.data_path)
    dataset_gamma = getattr(ds, "gamma", None)

    native_res = int(ds.domain_dimensions[0])
    res = args.res if args.res is not None else native_res
    print(
        f"Native resolution: {native_res}^3, output resolution: {res}^3, "
        f"field precision: {args.precision}, quantity type: {args.quantity_type}"
    )

    domain_left = ds.domain_left_edge.d
    domain_right = ds.domain_right_edge.d
    Lx = domain_right[0] - domain_left[0]

    all_data = ds.covering_grid(
        level=0, left_edge=domain_left, dims=[native_res] * 3
    )

    def parse_field(field_name):
        if isinstance(field_name, str) and ":" in field_name:
            ftype, fname = field_name.split(":", 1)
            return (ftype, fname)
        return field_name

    def get_native_field(field_name):
        # yt returns [x, y, z] order (Fortran); transpose to [z, y, x] = [k, j, i]
        return np.asarray(all_data[parse_field(field_name)].d.T, dtype=output_dtype)

    def get_field(field_name):
        data = get_native_field(field_name)
        if res != native_res:
            data = downsample(data, native_res, res)
        return np.ascontiguousarray(data, dtype=output_dtype)

    def to_output_grid(data):
        if res != native_res:
            data = downsample(data, native_res, res)
        return np.ascontiguousarray(data, dtype=output_dtype)

    def downsample(arr, from_res, to_res):
        factor = from_res // to_res
        if factor == 1:
            return arr
        # Volume averaging
        result = arr.reshape(to_res, factor, to_res, factor, to_res, factor)
        return result.mean(axis=(1, 3, 5))

    print("Reading density...")
    rho_native = get_native_field(args.rho_field)
    rho = to_output_grid(rho_native)

    if args.quantity_type == "conserved":
        print("Reading momentum density...")
        mom_x = get_field(args.mom_fields[0])
        mom_y = get_field(args.mom_fields[1])
        mom_z = get_field(args.mom_fields[2])
        print("Reading specific total energy...")
        specific_total_energy = get_native_field(args.total_energy_field)
        total_energy = to_output_grid(rho_native * specific_total_energy)
    else:
        print("Reading velocity...")
        vx = get_field(args.vel_fields[0])
        vy = get_field(args.vel_fields[1])
        vz = get_field(args.vel_fields[2])

    print("Reading magnetic field...")
    bx = get_field(args.mag_fields[0])
    by = get_field(args.mag_fields[1])
    bz = get_field(args.mag_fields[2])

    has_acc = not args.no_acc
    if has_acc:
        try:
            print("Reading acceleration...")
            ax = get_field(args.acc_fields[0])
            ay = get_field(args.acc_fields[1])
            az = get_field(args.acc_fields[2])
        except Exception as e:
            print(f"  Warning: acceleration not available ({e}), skipping")
            has_acc = False

    has_pres = args.quantity_type == "primitive" and not args.no_pres
    if has_pres:
        try:
            print("Reading pressure...")
            pres = get_field(args.pressure_field)
        except Exception as e:
            print(f"  Warning: pressure not available ({e}), skipping")
            has_pres = False

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
        if dataset_gamma is not None:
            stream.write_attribute("gamma", float(dataset_gamma))

        # Scalar fields
        stream.write("rho", rho, rho.shape, [0, 0, 0], rho.shape)

        if args.quantity_type == "conserved":
            stream.write("mom_x", mom_x, mom_x.shape, [0, 0, 0], mom_x.shape)
            stream.write("mom_y", mom_y, mom_y.shape, [0, 0, 0], mom_y.shape)
            stream.write("mom_z", mom_z, mom_z.shape, [0, 0, 0], mom_z.shape)
        else:
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

        if args.quantity_type == "conserved":
            stream.write(
                "total_energy",
                total_energy,
                total_energy.shape,
                [0, 0, 0],
                total_energy.shape,
            )
        elif has_pres:
            stream.write("pres", pres, pres.shape, [0, 0, 0], pres.shape)

        stream.end_step()

    print("Done.")
    if args.quantity_type == "conserved":
        print("  Fields written: rho, mom_{x,y,z}, mag_{x,y,z}, total_energy", end="")
    else:
        print("  Fields written: rho, vel_{x,y,z}, mag_{x,y,z}", end="")
    if has_acc:
        print(", acc_{x,y,z}", end="")
    if args.quantity_type == "primitive" and has_pres:
        print(", pres", end="")
    print()
    print(f"  Shape: ({res}, {res}, {res}) [k, j, i order]")
    print(f"  Field dtype: {np.dtype(output_dtype)}")
    if dataset_gamma is not None:
        print(f"  Gamma: {dataset_gamma}")
    print(f"  Domain: [{domain_left[0]}, {domain_right[0]}]^3, L = {Lx}")


if __name__ == "__main__":
    main()
