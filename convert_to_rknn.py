import json

from pathlib import Path

from rknn.api import RKNN


def main(args):
    """Convert the RetinaFace ONNX model to RKNN format with or without quantization.

    """
    # args.onnx_model_path.as_posix(), platform, do_quant, output_path = parse_arg()
    quant_suffix = "i8" if args.quantize else "fp"
    rknn_export_path = f"weights/{args.path_onnx_model.stem}_{quant_suffix}.rknn"
    custom_string = json.dumps({"mlflow_run_id": args.mlflow_run_id, "mlflow_run_name": args.mlflow_run_name})

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print("--> Config model")
    rknn.config(
        mean_values=[args.mean_values],
        std_values=[args.std_values],
        target_platform=args.platform,
        custom_string=custom_string,
    )
    print("done")

    # Load model
    print("--> Loading model")
    ret = rknn.load_onnx(model=args.path_onnx_model.as_posix())
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=args.quantize, dataset=args.path_dataset.as_posix())
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(rknn_export_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    # Release
    rknn.release()


if __name__ == "__main__":
    import argparse

    description = main.__doc__
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path_onnx_model", type=Path, help="Path to the ONNX model file.")
    parser.add_argument(
        "-ri",
        "--mlflow-run-id",
        type=str,
        default="26e2c3d9246240fc9d319917ec2f7db0",
        help="The MLFlow run-id associated with converted model.",
    )
    parser.add_argument(
        "-rn",
        "--mlflow-run-name",
        type=str,
        default="wise-jay-330",
        help="The MLFlow run-name associated with converted model.",
    )
    parser.add_argument("-q", "--quantize", action="store_true", help="Enable quantization of the RKNN model.")
    parser.add_argument(
        "-d",
        "--path-dataset",
        type=Path,
        default=Path("data/dataset.txt"),
        help="Path to list of image paths for quantization.",
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        required=False,
        default="rk3588",
        choices=["rk3562", "rk3566", "rk3568", "rk3588", "rk1808", "rv1109", "rv1126"],
        help="Target platform",
    )
    parser.add_argument("-m", "--mean_values", type=int, nargs="+", default=[104, 117, 123], help="Per-channel means.")
    parser.add_argument(
        "-s", "--std_values", type=int, nargs="+", default=[1, 1, 1], help="Per-channel standard deviations."
    )

    args = parser.parse_args()

    main(args)
