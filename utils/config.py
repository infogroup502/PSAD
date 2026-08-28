import os


MINIMAL_DEFAULTS = {
    "temporal_length": 20,
    "step": 1,
    "batch_size": 256,
    "num_workers": 0,
    "pin_memory": True,
    "num_epochs": 10,
    "pretrain_epochs": 0,
    "lr": 0.0002,
    "sensor_count": 0,
    "embed_dim": 128,
    "dropout": 0.1,
    "recon_hidden_dim": 0,
    "center_count": 80,
    "candidate_count": 3,
    "conv_channels": 0,
    "cluster_loss_weight": 1.0,
    "top_beta": 3,
    "dataset": "",
    "data_path": "",
    "index": 0,
    "threshold_ratio": 0.993,
    "threshold_source": "train",
    "extended_metrics_mode": "all",
    "periodic_point_diagnostics": False,
    "model_save_path": "checkpoints",
    "checkpoint_name": "",
    "result_root": "result",
    "result_dir": "",
    "result_csv_path": "",
    "run_name": "",
    "search_name": "",
    "grid_index": -1,
    "device": "cuda",
    "use_gpu": True,
    "enable_tf32": False,
    "cudnn_benchmark": False,
    "collapse_eps": 1e-6,
    "seed": 42,
}


def _as_float(cfg, key, default=0.0, lower=None):
    value = cfg.get(key, default)
    if value is None:
        value = default
    value = float(value)
    if lower is not None:
        value = max(float(lower), value)
    cfg[key] = value


def _as_int(cfg, key, default=0, lower=None):
    value = cfg.get(key, default)
    if value is None:
        value = default
    value = int(value)
    if lower is not None:
        value = max(int(lower), value)
    cfg[key] = value


def build_solver_config(config):

    cfg = dict(MINIMAL_DEFAULTS)
    for key in MINIMAL_DEFAULTS.keys():
        if key in config:
            cfg[key] = config[key]

    required = (
        "temporal_length",
        "batch_size",
        "num_epochs",
        "lr",
        "threshold_ratio",
        "center_count",
    )
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(f"Missing required command-line parameters: {', '.join(missing)}")
    _as_int(cfg, "center_count", 80, lower=1)
    _as_int(cfg, "candidate_count", 3, lower=1)
    cfg["candidate_count"] = min(
        int(cfg["candidate_count"]),
        int(cfg["center_count"]),
    )

    configured_channels = int(cfg.get("conv_channels", 0) or 0)
    cfg["conv_channels"] = (
        configured_channels
        if configured_channels > 0
        else max(1, int(cfg.get("embed_dim", 128)) // 2)
    )
    _as_float(cfg, "cluster_loss_weight", 1.0, lower=0.0)
    _as_int(cfg, "top_beta", 3, lower=1)

    _as_int(cfg, "pretrain_epochs", 0, lower=0)
    _as_int(cfg, "num_epochs", 3, lower=0)
    _as_int(cfg, "batch_size", 128, lower=1)
    _as_int(cfg, "step", 1, lower=1)
    _as_int(cfg, "num_workers", 0, lower=0)
    _as_float(cfg, "lr", 1e-4, lower=0.0)
    _as_float(cfg, "threshold_ratio", 0.995)
    if not 0.0 <= cfg["threshold_ratio"] <= 1.0:
        raise ValueError("threshold_ratio must be between 0 and 1")

    if cfg.get("temporal_length") is None:
        raise ValueError("temporal_length must be provided")
    _as_int(cfg, "temporal_length", lower=1)

    if not cfg.get("data_path"):
        cfg["data_path"] = cfg["dataset"]

    threshold_source = str(cfg.get("threshold_source", "train")).strip().lower()
    if threshold_source != "train":
        raise ValueError("clear mainline computes thresholds from train scores only.")
    cfg["threshold_source"] = "train"
    extended_metrics_mode = str(cfg.get("extended_metrics_mode", "all")).strip().lower()
    if extended_metrics_mode not in {"all", "none"}:
        raise ValueError(f"Unsupported extended_metrics_mode: {extended_metrics_mode}")
    cfg["extended_metrics_mode"] = extended_metrics_mode

    if not bool(cfg.get("use_gpu", True)):
        cfg["device"] = "cpu"

    if not cfg.get("result_dir"):
        cfg["result_dir"] = os.path.join(str(cfg.get("result_root", "result")), str(cfg["data_path"]))
    if not cfg.get("result_csv_path"):
        cfg["result_csv_path"] = os.path.join(cfg["result_dir"], f"{cfg['data_path']}.csv")
    if not cfg.get("model_save_path") or cfg.get("model_save_path") == "checkpoints":
        cfg["model_save_path"] = os.path.join(cfg["result_dir"], "checkpoints")

    cfg["pin_memory"] = bool(cfg.get("pin_memory", True))
    cfg["use_gpu"] = bool(cfg.get("use_gpu", True))
    cfg["enable_tf32"] = bool(cfg.get("enable_tf32", False))
    cfg["cudnn_benchmark"] = bool(cfg.get("cudnn_benchmark", False))
    _as_float(cfg, "collapse_eps", 1e-6, lower=0.0)
    _as_int(cfg, "seed", 42)

    return cfg
