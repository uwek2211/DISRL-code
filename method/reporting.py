from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from utils import overlap_scores


METRIC_FILES = {
    "nmi": "nmi_log.txt",
    "precision": "precision_log.txt",
    "recall": "recall_log.txt",
    "f1": "f1score_log.txt",
    "jaccard": "jaccard_log.txt",
}
TIME_FILES = {
    "seed_time": "seed_time_log.txt",
    "seed_avg_time": "seed_avg_time_log.txt",
}


def resolve_output_root(args, include_worker: bool = False) -> str:
    root = str(getattr(args, "output_root", "")).strip() or str(args.dataset_root)
    worker = str(getattr(args, "worker_tag", "")).strip()
    return os.path.join(root, worker) if include_worker and worker else root


def log_prefix(args) -> str:
    worker = str(getattr(args, "worker_tag", "")).strip()
    device = str(getattr(args, "device", "cpu"))
    return f"[{worker}|{device}]" if worker else f"[{device}]"


def _binary_membership(num_nodes: int, nodes: List[int]) -> np.ndarray:
    values = np.zeros(num_nodes, dtype=np.int32)
    valid = [int(node) for node in nodes if 0 <= int(node) < num_nodes]
    values[valid] = 1
    return values


def evaluate_prediction(detector, community_index: int, predicted_raw: List[int]):
    predicted = detector.graph.to_local_nodes(predicted_raw, allow_local=False)
    target = (
        detector.coms[community_index]
        if 0 <= community_index < len(detector.coms)
        else detector.graph.to_local_nodes(detector.raw_coms[community_index], allow_local=False)
    )
    y_pred = _binary_membership(detector.graph.n_nodes, predicted)
    y_true = _binary_membership(detector.graph.n_nodes, target)
    precision, recall, f1, jaccard = overlap_scores(predicted, target)
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


class RunReporter:
    def __init__(
        self,
        args,
        output_root: str,
        snapshots: List[str],
        seeds: List[int],
        metadata: Dict,
        clear_existing: bool = True,
    ):
        self.args = args
        self.output_root = output_root
        self.snapshots = snapshots
        self.seeds = [int(seed) for seed in seeds]
        self.save_dir = os.path.join(output_root, "saved_predictions")
        self.paths = {
            name: os.path.join(output_root, filename) for name, filename in {**METRIC_FILES, **TIME_FILES}.items()
        }
        self.scores = {name: {t: [] for t in range(len(snapshots))} for name in METRIC_FILES}
        self.snapshot_counts = {t: 0 for t in range(len(snapshots))}
        self.time_sum = {seed: 0.0 for seed in self.seeds}
        self.time_count = {seed: 0 for seed in self.seeds}
        self.metrics = {
            "by_seed": {seed: [] for seed in self.seeds},
            "by_snapshot": {snapshot: [] for snapshot in snapshots},
            "seed_time_avg": {},
            "seed_time_total": {},
            "seed_time_count": {},
            **metadata,
        }
        self._reset(clear_existing)

    def _reset(self, clear_existing: bool) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        if not clear_existing:
            return
        for path in [
            *self.paths.values(),
            os.path.join(self.output_root, "metrics.json"),
        ]:
            if os.path.exists(path):
                os.remove(path)
        for t in range(len(self.snapshots)):
            path = os.path.join(self.save_dir, f"pred_t{t:03d}.txt")
            if os.path.exists(path):
                os.remove(path)

    def record(
        self,
        detector,
        seed: int,
        t: int,
        community_index: int,
        predicted_raw: List[int],
        elapsed: float,
    ) -> None:
        snapshot = self.snapshots[t]
        values = evaluate_prediction(detector, community_index, predicted_raw)
        prediction_path = os.path.join(self.save_dir, f"pred_t{t:03d}.txt")
        with open(prediction_path, "a", encoding="utf-8") as handle:
            nodes = " ".join(map(str, sorted(set(map(int, predicted_raw)))))
            handle.write(f"{seed}: {nodes}\n")

        self.metrics["by_seed"][seed].append((snapshot, values))
        self.metrics["by_snapshot"][snapshot].append({"seed": seed, **values})
        print(
            f"{log_prefix(self.args)} Seed {seed:<5} | Snapshot "
            f"t{t:03d}/{len(self.snapshots) - 1:03d} | NMI: {values['nmi']:.4f} | "
            f"P: {values['precision']:.4f} | R: {values['recall']:.4f} | "
            f"F1: {values['f1']:.4f} | J: {values['jaccard']:.4f}",
            flush=True,
        )
        for name, value in values.items():
            with open(self.paths[name], "a", encoding="utf-8") as handle:
                handle.write(f"{seed} {t} {value:.4f}\n")
            self.scores[name][t].append(value)
        self.snapshot_counts[t] += 1
        self.time_sum[seed] += float(elapsed)
        self.time_count[seed] += 1
        with open(self.paths["seed_time"], "a", encoding="utf-8") as handle:
            handle.write(f"{seed} {t} {elapsed:.6f}\n")

    def finalize(self) -> Dict:
        for t, snapshot in enumerate(self.snapshots):
            averages = {
                name: float(np.mean(self.scores[name][t])) if self.scores[name][t] else 0.0 for name in METRIC_FILES
            }
            print(
                f"\n{log_prefix(self.args)} [Snapshot {snapshot}] done. Evaluated "
                f"{self.snapshot_counts[t]} seeds. Avg NMI: {averages['nmi']:.4f} | "
                f"Avg P: {averages['precision']:.4f} | Avg R: {averages['recall']:.4f} | "
                f"Avg F1: {averages['f1']:.4f} | Avg J: {averages['jaccard']:.4f}. "
                f"Pred dir: {self.save_dir}\n",
                flush=True,
            )

        with open(self.paths["seed_avg_time"], "w", encoding="utf-8") as handle:
            handle.write("seed avg_time_seconds total_time_seconds processed_snapshots\n")
            for seed in sorted(self.time_sum):
                total, count = self.time_sum[seed], self.time_count[seed]
                average = total / count if count else 0.0
                self.metrics["seed_time_avg"][seed] = average
                self.metrics["seed_time_total"][seed] = total
                self.metrics["seed_time_count"][seed] = count
                handle.write(f"{seed} {average:.6f} {total:.6f} {count}\n")

        _print_timing(self.time_sum, self.time_count, log_prefix(self.args))
        with open(os.path.join(self.output_root, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, ensure_ascii=False, indent=2)
        return self.metrics


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _parse_row(line: str, parse_elapsed: bool = False):
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        seed, snapshot = int(parts[0]), int(parts[1])
        return (seed, snapshot, float(parts[2]), line) if parse_elapsed else (seed, snapshot, line)
    except (TypeError, ValueError):
        return None


def _collect_worker_rows(output_root, worker_tags, relative_path, parser):
    rows, malformed = [], []
    for tag in worker_tags:
        for line in _read_lines(os.path.join(output_root, tag, relative_path)):
            row = parser(line)
            (rows if row else malformed).append(row or line)
    return rows, malformed


def _print_timing(time_sum: Dict[int, float], time_count: Dict[int, int], prefix: str) -> None:
    print(f"\n{prefix} Per-seed timing summary across snapshots:", flush=True)
    averages = []
    for seed in sorted(time_sum):
        total, count = float(time_sum[seed]), int(time_count[seed])
        average = total / count if count else 0.0
        if count:
            averages.append(average)
        print(
            f"{prefix} Seed {seed:<5} | Avg Time per Snapshot: {average:.4f}s | "
            f"Total Time: {total:.4f}s | Processed Snapshots: {count}",
            flush=True,
        )
    overall = float(np.mean(averages)) if averages else 0.0
    print(
        f"{prefix} Mean per-snapshot time across processed seeds: {overall:.4f}s",
        flush=True,
    )


def merge_worker_outputs(
    output_root: str,
    worker_tags: List[str],
    num_snapshots: int,
    clean_workers: bool = False,
) -> Dict[str, str]:
    os.makedirs(output_root, exist_ok=True)
    prediction_dir = os.path.join(output_root, "saved_predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    paths = {name: os.path.join(output_root, filename) for name, filename in {**METRIC_FILES, **TIME_FILES}.items()}
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)

    for name, filename in METRIC_FILES.items():
        parsed, malformed = _collect_worker_rows(output_root, worker_tags, filename, _parse_row)
        parsed.sort(key=lambda row: (row[0], row[1]))
        with open(paths[name], "w", encoding="utf-8") as handle:
            handle.writelines(f"{row[2]}\n" for row in parsed)
            handle.writelines(f"{line}\n" for line in malformed)

    time_rows, malformed = _collect_worker_rows(
        output_root, worker_tags, TIME_FILES["seed_time"], lambda line: _parse_row(line, parse_elapsed=True)
    )
    time_rows.sort(key=lambda row: (row[0], row[1]))
    with open(paths["seed_time"], "w", encoding="utf-8") as handle:
        handle.writelines(f"{row[3]}\n" for row in time_rows)
        handle.writelines(f"{line}\n" for line in malformed)

    time_sum, time_count = {}, {}
    for seed, _, elapsed, _ in time_rows:
        time_sum[seed] = time_sum.get(seed, 0.0) + elapsed
        time_count[seed] = time_count.get(seed, 0) + 1
    with open(paths["seed_avg_time"], "w", encoding="utf-8") as handle:
        handle.write("seed avg_time_seconds total_time_seconds processed_snapshots\n")
        for seed in sorted(time_sum):
            total, count = time_sum[seed], time_count[seed]
            handle.write(f"{seed} {total / count:.6f} {total:.6f} {count}\n")
    _print_timing(time_sum, time_count, "[Merge]")

    for t in range(num_snapshots):
        def parse_prediction(line):
            try:
                return int(line.split(":", 1)[0].strip()), line
            except ValueError:
                return None

        rows, malformed = _collect_worker_rows(
            output_root, worker_tags, os.path.join("saved_predictions", f"pred_t{t:03d}.txt"), parse_prediction
        )
        rows.sort(key=lambda row: row[0])
        target = os.path.join(prediction_dir, f"pred_t{t:03d}.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.writelines(f"{line}\n" for _, line in rows)
            handle.writelines(f"{line}\n" for line in malformed)

    if clean_workers:
        for tag in worker_tags:
            worker_root = os.path.join(output_root, tag)
            if os.path.isdir(worker_root):
                shutil.rmtree(worker_root)

    return {
        "output_root": output_root,
        **{f"{name}_log": paths[name] for name in METRIC_FILES},
        "seed_time_log": paths["seed_time"],
        "seed_avg_time_log": paths["seed_avg_time"],
        "pred_dir": prediction_dir,
    }
