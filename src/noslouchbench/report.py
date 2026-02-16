from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def aggregate_summaries(summary_dir: Path) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(summary_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        grouped[data["model_name"]].append(data)

    result: dict[str, dict] = {}
    for model_name, runs in grouped.items():
        n = len(runs)
        if n == 0:
            continue

        def avg(key: str) -> float:
            return sum(float(r[key]) for r in runs) / n

        result[model_name] = {
            "model_name": model_name,
            "num_sessions": n,
            "avg_duration_seconds": avg("duration_seconds"),
            "avg_fps_effective": avg("fps_effective"),
            "avg_detection_rate": avg("detection_rate"),
            "avg_slouch_ratio": avg("slouch_ratio"),
            "avg_upright_ratio": avg("upright_ratio"),
            "avg_latency_ms": avg("latency_ms_avg"),
            "avg_p95_latency_ms": avg("latency_ms_p95"),
        }

    return result


def write_reports(aggregated: dict[str, dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison.csv"
    md_path = output_dir / "model_comparison.md"

    rows = [aggregated[k] for k in sorted(aggregated)]
    fields = [
        "model_name",
        "num_sessions",
        "avg_duration_seconds",
        "avg_fps_effective",
        "avg_detection_rate",
        "avg_slouch_ratio",
        "avg_upright_ratio",
        "avg_latency_ms",
        "avg_p95_latency_ms",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        if not rows:
            f.write("No summary files found.\n")
        else:
            f.write("| Model | Sessions | Avg FPS | Avg Detection | Avg Slouch | Avg Latency (ms) | P95 Latency (ms) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    "| {model_name} | {num_sessions} | {avg_fps_effective:.2f} | {avg_detection_rate:.3f} | "
                    "{avg_slouch_ratio:.3f} | {avg_latency_ms:.2f} | {avg_p95_latency_ms:.2f} |\n".format(**row)
                )

    return csv_path, md_path

