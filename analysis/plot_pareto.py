#!/usr/bin/env python3
"""Plot Pareto frontier of throughput metrics across multiple series.

Usage:
    # Extract from job directories and plot:
    python analysis/plot_pareto.py \
        --series agg --dir 1234 1235 1236 \
        --series disagg_2p2d --dir 2345 2346 \
        --output pareto.png

    # Export data to JSON for later use:
    python analysis/plot_pareto.py \
        --series agg --dir 1234 1235 1236 \
        --export-dict analysis/my_data.json

    # Plot from saved data:
    python analysis/plot_pareto.py \
        --from-dict analysis/my_data.json \
        --output pareto.png

X-axis: Output tokens/s/user (latency-sensitive, higher = better for user experience)
Y-axis: Total output tokens/s/GPU (throughput-sensitive, higher = better for efficiency)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Default outputs directory for srtslurm jobs
SRTSLURM_OUTPUTS_DIR = Path("/lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm/outputs")


def find_srtslurm_job_dir(job_id: str, outputs_dir: Path = SRTSLURM_OUTPUTS_DIR) -> Path | None:
    """Find job directory by ID, handling both old (job_id) and new (job_id_config) formats.
    
    Also searches in subdirectories (e.g., outputs/1930500_agg_sweeps/1940819_*).
    """
    exact = outputs_dir / job_id
    if exact.exists():
        return exact
    
    # Search in top-level
    matches = list(outputs_dir.glob(f"{job_id}_*"))
    if matches:
        return matches[0]
    
    # Search in subdirectories
    matches = list(outputs_dir.glob(f"*/{job_id}_*"))
    if matches:
        return matches[0]
    
    return None


def find_aiperf_json(job_dir: Path) -> Path | None:
    """Find the profile_export_aiperf.json file for a job."""
    json_files = list(job_dir.glob("logs/artifacts/*/profile_export_aiperf.json"))
    json_files = [f for f in json_files if "warmup" not in str(f)]
    return json_files[0] if json_files else None


def find_config_yaml(job_dir: Path) -> Path | None:
    """Find the config.yaml file for a job."""
    config_path = job_dir / "logs" / "config.yaml"
    return config_path if config_path.exists() else None


def extract_metrics(job_dir: Path) -> Dict[str, object] | None:
    """Extract metrics from a job directory."""
    json_path = find_aiperf_json(job_dir)
    if not json_path:
        return None
    
    config_path = find_config_yaml(job_dir)
    gpus = None
    concurrency = None
    config_name = job_dir.name.split("_", 1)[1] if "_" in job_dir.name else None
    
    if config_path:
        try:
            import yaml
            with config_path.open() as f:
                config = yaml.safe_load(f)
            
            resources = config.get("resources", {})
            # Calculate total GPUs from config
            # For disaggregated: prefill_workers * gpus_per_prefill + decode_workers * gpus_per_decode
            # For aggregated: agg_workers * gpus_per_agg
            prefill_workers = resources.get("prefill_workers") or 0
            decode_workers = resources.get("decode_workers") or 0
            agg_workers = resources.get("agg_workers") or 0
            gpus_per_prefill = resources.get("gpus_per_prefill") or 0
            gpus_per_decode = resources.get("gpus_per_decode") or 0
            gpus_per_agg = resources.get("gpus_per_agg") or 0
            
            if agg_workers and gpus_per_agg:
                gpus = agg_workers * gpus_per_agg
            elif prefill_workers or decode_workers:
                gpus = (prefill_workers * gpus_per_prefill) + (decode_workers * gpus_per_decode)
            else:
                gpus = resources.get("gpus")
            
            benchmark = config.get("benchmark", {})
            concurrency = benchmark.get("concurrency") or benchmark.get("concurrencies")
            if not config_name:
                config_name = config.get("name")
            
            # Try to extract concurrency from config name (e.g., "conc8" -> 8)
            if not concurrency and config_name:
                import re
                match = re.search(r'conc(\d+)', config_name)
                if match:
                    concurrency = match.group(1)
        except Exception:
            pass
    
    try:
        with json_path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    
    output_tput_per_user = data.get("output_token_throughput_per_user", {}).get("avg")
    total_token_tput = data.get("total_token_throughput", {}).get("avg")
    ttft_p50 = data.get("time_to_first_token", {}).get("p50")

    total_token_tput_per_gpu = None
    if total_token_tput and gpus:
        total_token_tput_per_gpu = total_token_tput / gpus

    return {
        "job_id": job_dir.name.split("_")[0],
        "config_name": config_name,
        "concurrency": concurrency,
        "output_tput_per_user": output_tput_per_user,
        "total_token_tput": total_token_tput,
        "total_token_tput_per_gpu": total_token_tput_per_gpu,
        "ttft_p50": ttft_p50,
        "gpus": gpus,
    }


def fmt_2sf(value: float | None) -> str:
    """Format a value to 2 significant figures."""
    if value is None:
        return "?"
    if value == 0:
        return "0"
    mag = math.floor(math.log10(abs(value)))
    factor = 10 ** (mag - 1)
    rounded = round(value / factor) * factor
    return str(int(rounded))


def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute Pareto frontier for points where higher is better on both axes."""
    if not points:
        return []
    
    sorted_points = sorted(points, key=lambda p: -p[0])
    frontier = []
    max_y = float('-inf')
    
    for x, y in sorted_points:
        if y > max_y:
            frontier.append((x, y))
            max_y = y
    
    return sorted(frontier, key=lambda p: p[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontier of throughput metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default="pareto.png",
        help="Output file path (default: pareto.png)"
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(SRTSLURM_OUTPUTS_DIR),
        help=f"srtslurm outputs directory (default: {SRTSLURM_OUTPUTS_DIR})"
    )
    parser.add_argument(
        "--title",
        default="Throughput Pareto Frontier",
        help="Plot title"
    )
    parser.add_argument(
        "--no-frontier",
        action="store_true",
        help="Don't draw Pareto frontier lines"
    )
    parser.add_argument(
        "--label-points",
        action="store_true",
        help="Label each point with concurrency"
    )
    parser.add_argument(
        "--figsize",
        default="10,8",
        help="Figure size as 'width,height' (default: 10,8)"
    )
    parser.add_argument(
        "--export-dict",
        help="Export collected data to JSON file (can be loaded later with --from-dict)"
    )
    parser.add_argument(
        "--from-dict",
        help="Load data from JSON file instead of extracting from job directories"
    )
    
    # Parse series arguments manually since argparse doesn't handle repeated groups well
    return parser.parse_known_args()


def parse_series(remaining_args: List[str]) -> List[Tuple[str, List[str]]]:
    """Parse --series name --dir id1 id2 ... patterns from remaining args."""
    series_list = []
    current_series = None
    current_dirs = []
    
    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg == "--series":
            if current_series is not None:
                series_list.append((current_series, current_dirs))
            current_series = remaining_args[i + 1] if i + 1 < len(remaining_args) else "unnamed"
            current_dirs = []
            i += 2
        elif arg == "--dir":
            i += 1
            while i < len(remaining_args) and not remaining_args[i].startswith("--"):
                current_dirs.append(remaining_args[i])
                i += 1
        else:
            i += 1
    
    if current_series is not None:
        series_list.append((current_series, current_dirs))
    
    return series_list


def load_series_data_from_dict(dict_path: Path) -> Dict:
    """Load series data from a JSON file."""
    with dict_path.open() as f:
        data = json.load(f)
    
    # Convert points from lists back to tuples
    series_data = {}
    for series_name, series_info in data.get("series", {}).items():
        series_data[series_name] = {
            "points": [tuple(p) for p in series_info["points"]],
            "labels": series_info["labels"],
            "job_ids": series_info.get("job_ids", []),
            "ttft_p50_ms": series_info.get("ttft_p50_ms", []),
        }
    return series_data


def export_series_data_to_dict(series_data: Dict, dict_path: Path) -> None:
    """Export series data to a JSON file.
    
    Format is designed to be easy to edit manually:
    {
        "series": {
            "agg-tep8x3": {
                "points": [[x1, y1], [x2, y2], ...],
                "labels": ["c4", "c8", ...],
                "job_ids": ["1940824", "1940825", ...]
            },
            ...
        }
    }
    """
    export_data = {
        "series": {
            name: {
                "points": [list(p) for p in info["points"]],
                "labels": info["labels"],
                "job_ids": info.get("job_ids", []),
                "ttft_p50_ms": info.get("ttft_p50_ms", []),
            }
            for name, info in series_data.items()
        }
    }
    
    with dict_path.open("w") as f:
        json.dump(export_data, f, indent=2)
    print(f"Exported data to {dict_path}")


def collect_series_data(series_list: List[Tuple[str, List[str]]], outputs_dir: Path) -> Dict:
    """Collect data for each series from job directories."""
    series_data = {}
    for series_name, job_ids in series_list:
        points = []
        labels = []
        valid_job_ids = []
        ttft_p50_ms = []
        for job_id in job_ids:
            job_dir = find_srtslurm_job_dir(job_id, outputs_dir)
            if not job_dir:
                print(f"Warning: Job {job_id} not found", file=sys.stderr)
                continue
            
            metrics = extract_metrics(job_dir)
            if not metrics:
                print(f"Warning: Could not extract metrics from {job_id}", file=sys.stderr)
                continue
            
            x = metrics.get("output_tput_per_user")
            y = metrics.get("total_token_tput_per_gpu")
            
            if x is None or y is None:
                print(f"Warning: Missing metrics for {job_id} (x={x}, y={y})", file=sys.stderr)
                continue
            
            points.append((x, y))
            labels.append(f"c{metrics.get('concurrency', '?')}")
            valid_job_ids.append(job_id)
            ttft_p50_ms.append(metrics.get("ttft_p50"))

        if points:
            series_data[series_name] = {"points": points, "labels": labels, "job_ids": valid_job_ids, "ttft_p50_ms": ttft_p50_ms}
    
    return series_data


def main():
    args, remaining = parse_args()
    series_list = parse_series(remaining)
    
    outputs_dir = Path(args.outputs_dir)
    figsize = tuple(map(float, args.figsize.split(",")))
    
    # Load or collect series data
    if args.from_dict:
        dict_path = Path(args.from_dict)
        if not dict_path.exists():
            print(f"Error: Dict file not found: {dict_path}", file=sys.stderr)
            return 1
        series_data = load_series_data_from_dict(dict_path)
        print(f"Loaded data from {dict_path}")
    else:
        if not series_list:
            print("Error: No series specified. Use --series <name> --dir <job_ids>", file=sys.stderr)
            print("       Or use --from-dict <file.json> to load from saved data", file=sys.stderr)
            print("\nExample:", file=sys.stderr)
            print("  python plot_pareto.py --series agg --dir 1234 1235 --series disagg --dir 2345 2346", file=sys.stderr)
            return 1
        
        series_data = collect_series_data(series_list, outputs_dir)
    
    if not series_data:
        print("Error: No valid data points found", file=sys.stderr)
        return 1
    
    # Export if requested
    if args.export_dict:
        export_series_data_to_dict(series_data, Path(args.export_dict))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    
    for i, (series_name, data) in enumerate(series_data.items()):
        points = data["points"]
        labels = data["labels"]
        job_ids = data.get("job_ids", [])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        xs, ys = zip(*points)
        # Create legend label with job IDs in smaller text
        if job_ids:
            legend_label = f"{series_name}\n({', '.join(job_ids)})"
        else:
            legend_label = series_name
        ax.scatter(xs, ys, c=[color], marker=marker, s=80, label=legend_label, alpha=0.8)
        
        # Draw Pareto frontier
        if not args.no_frontier:
            frontier = compute_pareto_frontier(points)
            if len(frontier) > 1:
                fx, fy = zip(*frontier)
                ax.plot(fx, fy, c=color, linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Label points
        if args.label_points:
            ttft_p50_ms = data.get("ttft_p50_ms", [])
            for idx, ((x, y), label) in enumerate(zip(points, labels)):
                ttft = ttft_p50_ms[idx] if idx < len(ttft_p50_ms) else None
                if ttft is not None:
                    point_label = f"({label}, {fmt_2sf(ttft)})"
                else:
                    point_label = label
                ax.annotate(point_label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel("Output Tokens/s/User (higher = better latency)", fontsize=12)
    ax.set_ylabel("Total Output Tokens/s/GPU (higher = better throughput)", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.legend(loc="best", fontsize=9, title_fontsize=10)
    if args.label_points:
        ax.annotate("point labels: (concurrency, p50 TTFT ms)", xy=(1, 1), xycoords="axes fraction",
                    xytext=(-5, -5), textcoords="offset points", ha="right", va="top", fontsize=8,
                    color="gray", style="italic")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
    
    # Print summary table
    print("\nData summary:")
    print(f"{'Series':<20} {'Points':<8} {'X range':<20} {'Y range':<20}")
    print("-" * 70)
    for series_name, data in series_data.items():
        points = data["points"]
        xs, ys = zip(*points)
        print(f"{series_name:<20} {len(points):<8} {min(xs):.1f}-{max(xs):.1f}{'':>8} {min(ys):.1f}-{max(ys):.1f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
