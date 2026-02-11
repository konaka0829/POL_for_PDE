import argparse


DEFAULT_WAVELENGTH = 532e-9
DEFAULT_PIXEL_SIZE = 36e-6
DEFAULT_DISTANCE = 0.254


def add_one_optical_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--wavelength", type=float, default=DEFAULT_WAVELENGTH, help="Optical wavelength (meters).")
    parser.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE, help="Pixel size (meters).")
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE, help="Propagation distance (meters).")
    parser.add_argument(
        "--phase-init",
        choices=("uniform", "zero", "normal"),
        default="uniform",
        help="Initialization for phase-mask parameters.",
    )
    parser.add_argument(
        "--xbar-noise-std",
        type=float,
        default=0.0,
        help="Stddev of optional Gaussian noise for complex channel mixing output.",
    )
    parser.add_argument(
        "--donn-projection",
        choices=("power", "real", "magnitude"),
        default="power",
        help=(
            "Projection from complex optical field to real tensor. "
            "'power' matches third_party/ONE_PDE_public behavior."
        ),
    )
    parser.add_argument(
        "--donn-prop-padding",
        type=int,
        default=0,
        help="Zero-padding size used during Fresnel propagation inside DONN layer.",
    )
    parser.add_argument(
        "--domain-padding",
        type=int,
        default=9,
        help="Spatial padding used in the FNO-like block before operator application.",
    )
    parser.add_argument(
        "--one-mode",
        choices=("stagea", "tp_compat"),
        default="stagea",
        help="Model construction mode: Stage A baseline or third_party-compatible layout.",
    )
    parser.add_argument(
        "--donn-ratio",
        type=float,
        default=1.0,
        help="Scaling factor for DONN branch before residual merge (third_party uses this).",
    )
    return parser
