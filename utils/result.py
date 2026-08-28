import csv
import os
import time

import numpy as np


def print_score_stats(name, scores):
    print(
        f"{name} score min/max/mean/std: "
        f"{np.min(scores):.6f} / {np.max(scores):.6f} / "
        f"{np.mean(scores):.6f} / {np.std(scores):.6f}"
    )


def print_label_score_stats(scores, labels):
    labels = labels.astype(int)
    normal_scores = scores[labels == 0]
    abnormal_scores = scores[labels == 1]

    if len(normal_scores) > 0:
        print(
            "normal score mean/median: "
            f"{np.mean(normal_scores):.6f} / {np.median(normal_scores):.6f}"
        )
    else:
        print("normal score mean/median: nan / nan")

    if len(abnormal_scores) > 0:
        print(
            "abnormal score mean/median: "
            f"{np.mean(abnormal_scores):.6f} / {np.median(abnormal_scores):.6f}"
        )
    else:
        print("abnormal score mean/median: nan / nan")


def print_collapse_diagnostics(train_scores, test_scores, train_stats, test_stats, collapse_eps=1e-6):
    train_score_std = float(np.std(train_scores))
    test_score_std = float(np.std(test_scores))
    z_local_std = float(test_stats["z_local_std"])
    z_periodic_std = float(test_stats["z_periodic_std"])

    print(f"z_local std: {z_local_std:.6f}")
    print(f"z_periodic std: {z_periodic_std:.6f}")

    if train_score_std < collapse_eps or test_score_std < collapse_eps:
        print("[CollapseWarning] reconstruction score std is close to 0; scores may have collapsed.")
    if train_stats["z_local_std"] < collapse_eps or z_local_std < collapse_eps:
        print("[CollapseWarning] z_local std is close to 0; local encoder may have collapsed.")
    if train_stats["z_periodic_std"] < collapse_eps or z_periodic_std < collapse_eps:
        print("[CollapseWarning] z_periodic std is close to 0; periodic-point memory may have collapsed.")


def _csv_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_result_csv(result, config, result_dir, result_csv_path):
    os.makedirs(result_dir, exist_ok=True)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "run_name": config.get("run_name"),
        "search_name": config.get("search_name"),
        "grid_index": config.get("grid_index"),
        "dataset": config.get("dataset"),
        "data_path": config.get("data_path"),
        "index": config.get("index"),
        "checkpoint_name": config.get("checkpoint_name"),
        "model_save_path": config.get("model_save_path"),
        "sensor_count": config.get("sensor_count"),
        "temporal_length": config.get("temporal_length"),
        "step": config.get("step"),
        "center_count": config.get("center_count"),
        "candidate_count": config.get("candidate_count"),
        "conv_channels": config.get("conv_channels"),
        "top_beta": config.get("top_beta"),
        "spatial_kernel_size": int(config.get("sensor_count", 0) or 0) * int(config.get("temporal_length", 0) or 0),
        "temporal_kernel_size": config.get("temporal_length"),
        "reconstructor_route": "shared_2layer_mlp",
        "training_loss": "local_mse+periodic_point_mse+consistency_mse+cluster_aggregation",
        "topk_selection_route": "squared_euclidean_nearest",
        "periodic_point_fusion_route": "concat_linear",
        "topk_weight_route": "dot_similarity_softmax",
        "cluster_loss_route": "weighted_squared_distance",
        "cluster_loss_weight": config.get("cluster_loss_weight"),
        "score_route": "variable_topk_mean_local_periodic_diff",
        "batch_size": config.get("batch_size"),
        "num_epochs": config.get("num_epochs"),
        "pretrain_epochs": config.get("pretrain_epochs"),
        "lr": config.get("lr"),
        "threshold_ratio": result.get("threshold_ratio", config.get("threshold_ratio")),
        "threshold_source": config.get("threshold_source", "train"),
        "extended_metrics_mode": config.get("extended_metrics_mode", "all"),
        "threshold": result["threshold"],
        "score_mean": result.get("score_stats", {}).get("score_mean"),
        "score_std": result.get("score_stats", {}).get("score_std"),
        "score_max": result.get("score_stats", {}).get("score_max"),
        "pa_accuracy": result["pa_accuracy"],
        "pa_precision": result["pa_precision"],
        "pa_recall": result["pa_recall"],
        "pa_f_score": result["pa_f_score"],
        "VUS_ROC": result["VUS_ROC"],
        "VUS_PR": result["VUS_PR"],
    }

    write_header = not os.path.exists(result_csv_path) or os.path.getsize(result_csv_path) == 0
    fieldnames = list(row.keys())

    if not write_header:
        with open(result_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames
            existing_rows = list(reader)

        if existing_header:
            extra_fields = [key for key in row.keys() if key not in existing_header]
            fieldnames = existing_header + extra_fields
            if extra_fields:
                with open(result_csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(existing_rows)

    with open(result_csv_path, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

