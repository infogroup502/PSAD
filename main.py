import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn

from solver import Solver


warnings.filterwarnings("ignore")


BASE_CONFIG = {
    "seed": 42,
    "mode": "train",
    "device": "cuda",
    "use_gpu": True,
    "enable_tf32": False,
    "cudnn_benchmark": False,
    "deterministic": True,
    "num_workers": 0,
    "pin_memory": True,
    "log_to_file": True,
    "dataset": "",
    "data_path": "",
    "index": 0,
    "sensor_count": 0,
    "temporal_length": 20,
    "step": 1,
    "batch_size": 256,
    "num_epochs": 10,
    "pretrain_epochs": 0,
    "lr": 0.0002,
    "threshold_ratio": 0.993,
    "threshold_source": "train",
    "extended_metrics_mode": "all",
    "periodic_point_diagnostics": False,
    "embed_dim": 128,
    "dropout": 0.1,
    "recon_hidden_dim": 0,
    "center_count": 80,
    "candidate_count": 3,
    "conv_channels": 0,
    "cluster_loss_weight": 1.0,
    "top_beta": 3,
    "model_save_path": "checkpoints",
    "checkpoint_name": "",
    "result_root": "result",
    "result_dir": "",
    "result_csv_path": "",
    "log_path": "",
    "run_name": "",
    "search_name": "",
    "grid_index": -1,
    "collapse_eps": 1e-6,
    "train_only": False,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


class SummaryLogger:
    SUMMARY_PREFIXES = (
        "train score min/max/mean/std:",
        "test score min/max/mean/std:",
        "normal score mean/median:",
        "abnormal score mean/median:",
        "z_local std:",
        "z_periodic std:",
        "threshold_ratio:",
        "Threshold:",
        "PA F-score:",
        "Raw Accuracy",
        "PA  Accuracy",
        "pa_accuracy",
        "pa_precision",
        "pa_recall",
        "pa_f_score",
        "MCC_score",
        "Affiliation precision",
        "Affiliation recall",
        "R_AUC_ROC",
        "R_AUC_PR",
        "VUS_ROC",
        "VUS_PR",
        "[Result]",
        "Scoring train:",
        "Scoring test:",
    )

    def __init__(self, filename, stream):
        self.terminal = stream
        self.filename = filename
        self.buffer = ""
        self.in_hyperparameters = False

    def write(self, message):
        self.terminal.write(message)
        for char in message or "":
            if char in "\r\n":
                self._write_summary_line(self.buffer)
                self.buffer = ""
            else:
                self.buffer += char

    def _write_summary_line(self, line):
        text = line.strip()
        if not text:
            return

        should_write = False
        if len(text) == 19 and text[4] == "-" and text[7] == "-" and text[10] == " ":
            should_write = True
        elif text == "================ PSAD Configuration ===============":
            self.in_hyperparameters = True
            should_write = True
        elif text.startswith("========== PSAD reconstruction evaluation"):
            self.in_hyperparameters = False
            should_write = True
        elif text.startswith("================"):
            self.in_hyperparameters = False
        elif self.in_hyperparameters and ":" in text:
            should_write = True
        elif text.startswith(("Scoring train:", "Scoring test:")):
            should_write = "100%" in text
        elif text.startswith(self.SUMMARY_PREFIXES):
            should_write = True

        if should_write:
            with open(self.filename, "a+", encoding="utf-8") as log:
                log.write(text + "\n")

    def flush(self):
        self.terminal.flush()


def set_seed(seed, deterministic=True):
    deterministic = bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="PSAD multivariate time-series anomaly detection."
    )

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["train", "test"])

    parser.add_argument("--temporal_length", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=None)
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--lr", dest="lr", type=float, default=None)
    parser.add_argument("--threshold_ratio", type=float, default=None)
    parser.add_argument("--threshold_source", type=str, default=None, choices=["train"])
    parser.add_argument("--extended_metrics_mode", type=str, default=None, choices=["all", "none"])
    parser.add_argument("--periodic_point_diagnostics", type=str2bool, default=None)

    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--recon_hidden_dim", type=int, default=None)
    parser.add_argument("--center_count", type=int, default=None)
    parser.add_argument("--candidate_count", type=int, default=None)
    parser.add_argument("--conv_channels", type=int, default=None)

    parser.add_argument("--cluster_loss_weight", type=float, default=None)
    parser.add_argument("--top_beta", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=str2bool, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_gpu", type=str2bool, default=None)
    parser.add_argument("--enable_tf32", type=str2bool, default=None)
    parser.add_argument("--cudnn_benchmark", type=str2bool, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--result_csv_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--search_name", type=str, default=None)
    parser.add_argument("--grid_index", type=int, default=None)
    parser.add_argument("--collapse_eps", type=float, default=None)
    parser.add_argument("--log_to_file", type=str2bool, default=None)
    parser.add_argument("--train_only", type=str2bool, default=None)
    return parser


def _default_checkpoint_name(config, include_timestamp=False):
    dataset = str(config["dataset"])
    center_count = int(config["center_count"])
    candidate_count = int(config.get("candidate_count", 3))
    topk_suffix = "" if candidate_count == 3 else f"_top{candidate_count}"
    channels = int(config.get("conv_channels", 0) or 0)
    stem = (
        f"psad_{dataset}_periodic_points_"
        f"c{channels}_k{center_count}{topk_suffix}"
    )
    if include_timestamp:
        stem = f"{stem}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    return f"{stem}.pt"


def _validate_mainline_args(config):
    threshold_source = str(config.get("threshold_source", "train")).strip().lower()
    if threshold_source != "train":
        raise ValueError("clear mainline computes thresholds from train scores only; use threshold_source=train.")
    config["threshold_source"] = "train"


def build_runtime_config(args):
    config = dict(BASE_CONFIG)
    config["dataset"] = args.dataset

    for key in [
        "data_path",
        "index",
        "mode",
        "temporal_length",
        "step",
        "batch_size",
        "num_epochs",
        "pretrain_epochs",
        "lr",
        "threshold_ratio",
        "threshold_source",
        "extended_metrics_mode",
        "periodic_point_diagnostics",
        "embed_dim",
        "dropout",
        "recon_hidden_dim",
        "center_count",
        "candidate_count",
        "conv_channels",
        "cluster_loss_weight",
        "top_beta",
        "num_workers",
        "pin_memory",
        "device",
        "use_gpu",
        "enable_tf32",
        "cudnn_benchmark",
        "deterministic",
        "seed",
        "model_save_path",
        "checkpoint_name",
        "result_root",
        "result_dir",
        "result_csv_path",
        "log_path",
        "run_name",
        "search_name",
        "grid_index",
        "collapse_eps",
        "log_to_file",
        "train_only",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    _validate_mainline_args(config)

    if not config["data_path"]:
        config["data_path"] = config["dataset"]
    if not bool(config["use_gpu"]):
        config["device"] = "cpu"

    config["cluster_loss_weight"] = max(0.0, float(config.get("cluster_loss_weight", 1.0) or 0.0))

    configured_channels = int(config.get("conv_channels", 0) or 0)
    config["conv_channels"] = (
        configured_channels
        if configured_channels > 0
        else max(1, int(config["embed_dim"]) // 2)
    )
    config["candidate_count"] = min(
        max(1, int(config.get("candidate_count", 3) or 3)),
        int(config["center_count"]),
    )
    config["top_beta"] = max(1, int(config.get("top_beta", 3) or 3))
    config["batch_size"] = int(config["batch_size"])
    config["deterministic"] = bool(config.get("deterministic", True))
    if not config.get("result_dir"):
        config["result_dir"] = os.path.join(config["result_root"], str(config["data_path"]))
    if not config.get("log_path"):
        config["log_path"] = os.path.join(config["result_dir"], f"{config['data_path']}.log")
    if not config.get("result_csv_path"):
        config["result_csv_path"] = os.path.join(config["result_dir"], f"{config['data_path']}.csv")
    if args.model_save_path is None:
        config["model_save_path"] = os.path.join(config["result_dir"], "checkpoints")
    if not config.get("checkpoint_name") and config["mode"] == "train":
        config["checkpoint_name"] = _default_checkpoint_name(config, include_timestamp=True)

    return config


def run(config):
    cudnn.benchmark = bool(config["cudnn_benchmark"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    solver = Solver(config)
    if config["mode"] == "train":
        solver.train()
        if not bool(config.get("train_only", False)):
            solver.test(load_model=False)
    else:
        solver.test(load_model=True)
    return solver


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    config = build_runtime_config(args)
    set_seed(config["seed"], deterministic=config.get("deterministic", True))

    if config["log_to_file"]:
        os.makedirs(config["result_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(config["log_path"]) or ".", exist_ok=True)
        sys.stdout = SummaryLogger(config["log_path"], stream=sys.stdout)
        sys.stderr = SummaryLogger(config["log_path"], stream=sys.stderr)

    print("\n\n")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("================ PSAD Configuration ===============")
    display_keys = (
        "dataset",
        "data_path",
        "mode",
        "device",
        "temporal_length",
        "batch_size",
        "num_epochs",
        "pretrain_epochs",
        "lr",
        "threshold_ratio",
        "center_count",
        "candidate_count",
        "conv_channels",
        "cluster_loss_weight",
        "seed",
        "checkpoint_name",
        "result_dir",
    )
    for key in display_keys:
        value = config[key]
        print(f"{key}: {value}")

    return run(config)


if __name__ == "__main__":
    main()
