import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from data_factory.data_loader import get_loader_segment
from model.psad import PSADModel
from utils.config import build_solver_config
from utils.loss import (
    cluster_aggregation_loss,
    reconstruction_loss,
    training_loss,
)
from utils.periodic_point import (
    print_periodic_point_usage_diagnostics,
    print_periodic_point_initialization,
)
from utils.result import write_result_csv
from utils.score import (
    score_from_reconstruction,
    target_from_window,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class IsolatedShuffleSampler(RandomSampler):

    def __iter__(self):
        if self.generator is not None:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        yield from super().__iter__()


def progress_bar(iterable, **kwargs):
    if os.environ.get("PSAD_DISABLE_TQDM", "0") == "1":
        return iterable
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


class Solver(object):

    def __init__(self, config):
        cfg = build_solver_config(config)
        self.config = cfg
        self.__dict__.update(cfg)

        self.device = self._build_device()
        self._configure_torch_backends()
        self._build_loaders()
        self.sensor_count = self._resolve_sensor_count()
        self.config["sensor_count"] = self.sensor_count
        self._build_model()

    def _build_device(self):
        requested = str(getattr(self, "device", "cuda") or "cuda").strip().lower()
        if not bool(getattr(self, "use_gpu", True)):
            requested = "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but the active Python/PyTorch environment has no CUDA support. "
                "Activate a CUDA-enabled environment or run with --device cpu --use_gpu false."
            )
        return torch.device(requested)

    def _configure_torch_backends(self):
        if self.device.type != "cuda":
            return
        allow_tf32 = bool(getattr(self, "enable_tf32", False))
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = bool(getattr(self, "cudnn_benchmark", False))
        except Exception:
            pass

    def _loader_window_size(self):

        return int(self.temporal_length) + 1

    def _build_loaders(self):
        data_root = os.path.join("dataset", self.data_path)
        loader_window_size = self._loader_window_size()
        train_loader = get_loader_segment(
            self.index,
            data_root,
            batch_size=self.batch_size,
            win_size=loader_window_size,
            step=self.step,
            mode="train",
            dataset=self.dataset,
        )
        test_loader = get_loader_segment(
            self.index,
            data_root,
            batch_size=self.batch_size,
            win_size=loader_window_size,
            step=self.step,
            mode="test",
            dataset=self.dataset,
        )
        self.train_loader = self._wrap_loader(train_loader, shuffle=True)
        self.test_loader = self._wrap_loader(test_loader, shuffle=False)

    def _wrap_loader(self, loader, shuffle):
        num_workers = max(0, int(self.num_workers))
        pin_memory = bool(self.pin_memory) and self.device.type == "cuda"
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(self.seed))
        kwargs = {
            "dataset": loader.dataset,
            "batch_size": int(self.batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "drop_last": False,
            "pin_memory": pin_memory,
            "generator": loader_generator,
        }
        if shuffle:
            shuffle_generator = torch.Generator()
            shuffle_generator.manual_seed(int(self.seed))
            kwargs["sampler"] = IsolatedShuffleSampler(
                loader.dataset,
                generator=shuffle_generator,
            )
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(**kwargs)

    def _to_device(self, tensor):
        return tensor.to(
            self.device,
            dtype=torch.float32,
            non_blocking=bool(self.pin_memory) and self.device.type == "cuda",
        )

    def _embedding_std_tensor(self, z):
        z = z.detach().reshape(-1, z.size(-1))
        if z.size(0) <= 1:
            return z.new_tensor(0.0)
        return z.std(dim=0, unbiased=False).mean()

    def _new_epoch_stats(self):
        return {
            "batches": 0,
            "score_count": 0,
            "loss_sum": torch.zeros((), device=self.device),
            "cluster_loss_sum": torch.zeros((), device=self.device),
            "score_sum": torch.zeros((), device=self.device),
            "score_sumsq": torch.zeros((), device=self.device),
            "z_local_std_sum": torch.zeros((), device=self.device),
            "z_periodic_std_sum": torch.zeros((), device=self.device),
        }

    def _update_score_stats(self, stats, score):
        score_detached = score.detach()
        stats["score_count"] += int(score_detached.numel())
        stats["score_sum"] = stats["score_sum"] + score_detached.sum()
        stats["score_sumsq"] = stats["score_sumsq"] + (score_detached * score_detached).sum()

    def _finalize_epoch_stats(self, stats):
        batches = max(1, int(stats["batches"]))
        count = max(1, int(stats["score_count"]))
        score_mean = stats["score_sum"] / float(count)
        score_var = stats["score_sumsq"] / float(count) - score_mean * score_mean
        score_std = torch.sqrt(torch.clamp(score_var, min=0.0))
        result = {
            "loss": stats["loss_sum"] / float(batches),
            "cluster_loss": stats["cluster_loss_sum"] / float(batches),
            "score_mean": score_mean,
            "score_std": score_std,
            "z_local_std": stats["z_local_std_sum"] / float(batches),
            "z_periodic_std": stats["z_periodic_std_sum"] / float(batches),
        }
        return {key: float(value.detach().cpu().item()) for key, value in result.items()}

    def _new_score_stats(self):
        return {
            "batches": 0,
            "z_local_std_sum": torch.zeros((), device=self.device),
            "z_periodic_std_sum": torch.zeros((), device=self.device),
        }

    def _finalize_score_stats(self, stats):
        batches = max(1, int(stats["batches"]))
        return {
            "z_local_std": float((stats["z_local_std_sum"] / float(batches)).detach().cpu().item()),
            "z_periodic_std": float((stats["z_periodic_std_sum"] / float(batches)).detach().cpu().item()),
        }

    def _resolve_sensor_count(self):
        inferred = None
        dataset = getattr(self.train_loader, "dataset", None)
        train_data = getattr(dataset, "train", None)
        if train_data is not None and hasattr(train_data, "shape") and len(train_data.shape) == 2:
            inferred = int(train_data.shape[1])

        configured = int(self.sensor_count) if self.sensor_count is not None else 0
        if inferred is None:
            if configured <= 0:
                raise ValueError("sensor_count must be set when it cannot be inferred from data")
            return configured

        if configured > 0 and configured != inferred:
            print(f"[Config] sensor_count={configured} differs from data sensors={inferred}; using data sensors")
        return inferred

    def _build_model(self):
        self.model = PSADModel(
            sensor_count=self.sensor_count,
            embed_dim=self.embed_dim,
            temporal_length=self.temporal_length,
            dropout=self.dropout,
            recon_hidden_dim=self.recon_hidden_dim,
            center_count=self.center_count,
            candidate_count=self.candidate_count,
            conv_channels=self.conv_channels,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _checkpoint_path(self):
        checkpoint_name = self.checkpoint_name
        if not checkpoint_name:
            checkpoint_name = f"{self._checkpoint_name_prefix()}.pt"
        return os.path.join(self.model_save_path, checkpoint_name)

    def _checkpoint_name_prefix(self):
        topk_suffix = "" if int(self.candidate_count) == 3 else f"_top{int(self.candidate_count)}"
        channels = int(self.conv_channels)
        return (
            f"psad_{self.dataset}_periodic_points_"
            f"c{channels}_k{int(self.center_count)}{topk_suffix}"
        )

    def _latest_checkpoint_path(self):
        if self.checkpoint_name:
            return self._checkpoint_path()
        prefix = self._checkpoint_name_prefix()
        if not os.path.isdir(self.model_save_path):
            return self._checkpoint_path()
        candidates = []
        for name in os.listdir(self.model_save_path):
            if name.startswith(prefix) and name.endswith(".pt"):
                path = os.path.join(self.model_save_path, name)
                candidates.append((os.path.getmtime(path), path))
        if not candidates:
            return self._checkpoint_path()
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _save_checkpoint(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        torch.save(self.model.state_dict(), self._checkpoint_path())

    def load_checkpoint(self):
        path = self._latest_checkpoint_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _make_local_view(self, x):
        return x

    def _forward_loss(self, x, include_cluster=True):
        x_local = self._make_local_view(x)
        target = target_from_window(x)
        z_local, z_periodic, xhat_local, xhat_periodic = self.model(x_local)

        local_loss = reconstruction_loss(xhat_local, target)
        periodic_point_loss = reconstruction_loss(xhat_periodic, target)
        consistency_loss = reconstruction_loss(xhat_local, xhat_periodic)

        cluster_loss = x.new_tensor(0.0)
        if bool(include_cluster):
            cluster_loss = cluster_aggregation_loss(
                z_local,
                self.model.periodic_point_memory.last_candidate_points,
                self.model.periodic_point_memory.last_candidate_weights,
            )

        loss = training_loss(
            local_loss,
            periodic_point_loss,
            consistency_loss,
            cluster_loss,
            cluster_loss_weight=float(self.cluster_loss_weight),
        )
        score = score_from_reconstruction(
            xhat_local,
            xhat_periodic,
            top_beta=self.top_beta,
        )
        return loss, score, z_local, z_periodic, cluster_loss

    def _forward_local_pretrain_loss(self, x):
        x_local = self._make_local_view(x)
        target = target_from_window(x)
        z_local = self.model.encode_local(x_local)
        xhat_local = self.model.reconstruct(z_local)
        local_loss = reconstruction_loss(xhat_local, target)
        score = ((xhat_local - target) ** 2).max(dim=-1).values
        return local_loss, score, z_local

    def _pretrain_encoder(self):
        total_epochs = int(self.pretrain_epochs)
        if total_epochs <= 0:
            return

        for _ in range(total_epochs):
            self.model.train()
            for input_data, _ in self.train_loader:
                x = self._to_device(input_data)
                self.optimizer.zero_grad()
                loss, _, _ = self._forward_local_pretrain_loss(x)
                loss.backward()
                self.optimizer.step()

    def _train_one_epoch(self):
        self.model.train()
        for input_data, _ in self.train_loader:
            x = self._to_device(input_data)

            self.optimizer.zero_grad()
            loss, _, _, _, _ = self._forward_loss(x)
            loss.backward()
            self.optimizer.step()

    def train(self):
        self._pretrain_encoder()
        print_periodic_point_initialization(self.model)
        print("\n========== PSAD reconstruction training ==========")

        epoch_iter = progress_bar(
            range(int(self.num_epochs)),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
        for _ in epoch_iter:
            self._train_one_epoch()

        self._save_checkpoint()
        if bool(self.periodic_point_diagnostics):
            print_periodic_point_usage_diagnostics(
                model=self.model,
                train_loader=self.train_loader,
                device=self.device,
                center_count=self.center_count,
                make_local_view=self._make_local_view,
                progress_bar=progress_bar,
            )

    @torch.no_grad()
    def score_loader(self, loader, return_labels=False, desc="score"):
        self.model.eval()
        all_scores = []
        all_labels = []
        stats = self._new_score_stats()

        for input_data, labels in loader:
            x = self._to_device(input_data)
            _, score, z_local, z_periodic, _ = self._forward_loss(x, include_cluster=False)
            all_scores.append(score.detach().to("cpu", non_blocking=True))
            stats["batches"] += 1
            stats["z_local_std_sum"] = stats["z_local_std_sum"] + self._embedding_std_tensor(z_local)
            stats["z_periodic_std_sum"] = stats["z_periodic_std_sum"] + self._embedding_std_tensor(z_periodic)

            if return_labels:
                label_values = labels[:, -1]
                all_labels.append(label_values.detach().to("cpu", non_blocking=True))

        scores = torch.cat(all_scores, dim=0).numpy().reshape(-1)
        labels = None
        if return_labels:
            labels = torch.cat(all_labels, dim=0).numpy().reshape(-1)

        return scores, labels, self._finalize_score_stats(stats)

    def _evaluate_predictions(self, test_scores, test_labels, threshold):
        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)
        from metrics.metrics import calculate_selected_scores

        return calculate_selected_scores(pred, gt)

    def _print_evaluation_result(self, result):
        for key in (
            "pa_accuracy",
            "pa_precision",
            "pa_recall",
            "pa_f_score",
            "VUS_ROC",
            "VUS_PR",
        ):
            value = result[key]
            print("{0:21} : {1:0.4f}".format(key, value))

    def test(self, load_model=False):
        if load_model:
            self.load_checkpoint()

        score_target = "last-point"
        print(f"\n========== PSAD reconstruction evaluation ({score_target} score) ==========")
        train_scores, _, train_stats = self.score_loader(
            self.train_loader,
            return_labels=False,
            desc="Scoring train",
        )
        test_scores, test_labels, test_stats = self.score_loader(
            self.test_loader,
            return_labels=True,
            desc="Scoring test",
        )

        score_stats = {
            "score_mean": float(np.mean(test_scores)),
            "score_std": float(np.std(test_scores)),
            "score_max": float(np.max(test_scores)),
        }
        os.makedirs(self.result_dir, exist_ok=True)
        np.savez(
            os.path.join(self.result_dir, "psad_scores.npz"),
            train_scores=np.asarray(train_scores),
            test_scores=np.asarray(test_scores),
            test_labels=np.asarray(test_labels).astype(int),
        )
        threshold = np.quantile(train_scores, float(self.threshold_ratio))
        result = self._evaluate_predictions(
            test_scores,
            test_labels,
            threshold,
        )
        result["score_stats"] = score_stats
        result["threshold"] = threshold
        result["threshold_ratio"] = float(self.threshold_ratio)

        self._print_evaluation_result(result)
        write_result_csv(
            result=result,
            config=self.config,
            result_dir=self.result_dir,
            result_csv_path=self.result_csv_path,
        )
        return result
