import argparse


def add_data_mode_args(
    parser: argparse.ArgumentParser,
    *,
    default_data_mode: str,
    default_data_file: str,
    default_train_file: str | None = None,
    default_test_file: str | None = None,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--data-mode",
        choices=("single_split", "separate_files"),
        default=default_data_mode,
        help="Data loading mode: single file split into train/test or separate files.",
    )
    parser.add_argument(
        "--data-file",
        default=default_data_file,
        help="Single data file (used when --data-mode=single_split).",
    )
    parser.add_argument(
        "--train-file",
        default=default_train_file,
        help="Training data file (used when --data-mode=separate_files).",
    )
    parser.add_argument(
        "--test-file",
        default=default_test_file,
        help="Test data file (used when --data-mode=separate_files).",
    )
    return parser


def add_split_args(
    parser: argparse.ArgumentParser,
    *,
    default_train_split: float = 0.8,
    default_seed: int = 0,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--train-split",
        type=float,
        default=default_train_split,
        help="Fraction of samples to use for training in single_split mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Random seed for reproducible train/test splits.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before splitting in single_split mode.",
    )
    return parser


def validate_data_mode_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.data_mode == "single_split":
        if not args.data_file:
            parser.error("--data-file is required when --data-mode=single_split")
        if hasattr(args, "train_split") and not (0.0 < args.train_split < 1.0):
            parser.error("--train-split must be in the interval (0, 1)")
    else:
        if not args.train_file or not args.test_file:
            parser.error("--train-file and --test-file are required when --data-mode=separate_files")
