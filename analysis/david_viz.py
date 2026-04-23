#!/usr/bin/env python3
"""Parse AIPerf trace-replay summary artifacts from one or more run folders.

Usage:
    # Parse from srtslurm job IDs (looks in outputs/ directory)
    python david_viz.py --dir 1930535 1930536
    
    # Parse from direct paths
    python david_viz.py /path/to/run/folder
    
    # Output TSV for Excel paste (tab-separated, prints to stdout)
    python david_viz.py --dir 1930535 --tsv
    
    # Print just the header
    python david_viz.py --header
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# Default outputs directory for srtslurm jobs
SRTSLURM_OUTPUTS_DIR = Path("/lustre/fsw/coreai_dlfw_dev/karenc/srt-slurm/outputs")

METRIC_TTFT = "Time to First Token (ms)"
METRIC_OUTPUT_TPUT_PER_USER = "Output Token Throughput Per User (tokens/sec/user)"
METRIC_TOTAL_TOKEN_TPUT = "Total Token Throughput (tokens/sec)"
METRIC_REQUEST_TPUT = "Request Throughput (requests/sec)"
METRIC_REQUEST_COUNT = "Request Count"
METRIC_ERROR_REQUEST_COUNT = "Error Request Count"
SLA_TTFT_MS = 3000.0
SLA_ITL_MS = 8.0

# Columns for TSV/Excel output
TSV_COLUMNS = [
    "dataset",
    "srtslurm_id",
    "config_name",
    "concurrency",
    "request_count",
    "errors [pct]",
    "runtime_error",
    "ttft_avg_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_avg_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "output_tput_per_user",
    "total_token_tput",
    "total_token_tput_per_gpu",
    "request_tput",
    "goodput",
    "kv_total_blocks (dynamo_component_total_blocks)",
    "kv_blocksize (dynamo_frontend_model_kv_cache_block_size)",
    "kv_total_workspace_GiB (calculated)",
    "kv_util_max (trtllm_kv_cache_utilization)",
    "kv_reused_blocks (trtllm_kv_cache_reused_blocks)",
    "kv_missed_blocks (trtllm_kv_cache_missed_blocks)",
    "kv_hit_rate (calculated)",
]


def load_metrics(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open(newline="") as f:
        return {row["Metric"]: row for row in csv.DictReader(f)}


def find_srtslurm_job_dir(job_id: str, outputs_dir: Path = SRTSLURM_OUTPUTS_DIR) -> Path | None:
    """Find job directory by ID, handling both old (job_id) and new (job_id_config) formats."""
    # Try exact match first
    exact = outputs_dir / job_id
    if exact.exists():
        return exact
    
    # Try glob for job_id_* pattern
    matches = list(outputs_dir.glob(f"{job_id}_*"))
    if matches:
        return matches[0]
    
    return None


def extract_srtslurm_info(job_dir: Path) -> Dict[str, object]:
    """Extract srtslurm-specific info from a job directory."""
    info = {
        "srtslurm_id": "",
        "config_name": "",
        "dataset": "",
        "concurrency": None,
        "gpus": 8,  # default
    }
    
    # Parse job ID and config name from directory name
    dir_name = job_dir.name
    parts = dir_name.split("_", 1)
    info["srtslurm_id"] = parts[0]
    if len(parts) > 1:
        info["config_name"] = parts[1]
    
    # Try to load config.yaml for more details
    config_file = job_dir / "logs" / "config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
                if not info["config_name"]:
                    info["config_name"] = config.get("name", "")
                info["concurrency"] = config.get("benchmark", {}).get("concurrencies")
                trace_file = config.get("benchmark", {}).get("trace_file", "")
                if trace_file:
                    info["dataset"] = Path(trace_file).parent.name
                # Calculate GPUs from resources
                resources = config.get("resources", {})
                agg_workers = resources.get("agg_workers")
                gpus_per_agg = resources.get("gpus_per_agg")
                if agg_workers and gpus_per_agg:
                    info["gpus"] = agg_workers * gpus_per_agg
                else:
                    prefill_workers = resources.get("prefill_workers") or 0
                    gpus_per_prefill = resources.get("gpus_per_prefill") or 0
                    decode_workers = resources.get("decode_workers") or 0
                    gpus_per_decode = resources.get("gpus_per_decode") or 0
                    total = prefill_workers * gpus_per_prefill + decode_workers * gpus_per_decode
                    if total > 0:
                        info["gpus"] = total
        except Exception:
            pass
    
    return info


def find_srtslurm_aiperf_json(job_dir: Path) -> Path | None:
    """Find the profile_export_aiperf.json file for a srtslurm job."""
    # Look in logs/artifacts/*/profile_export_aiperf.json (exclude warmup)
    json_files = list(job_dir.glob("logs/artifacts/*/profile_export_aiperf.json"))
    json_files = [f for f in json_files if "warmup" not in str(f)]
    
    if json_files:
        return json_files[0]
    return None


def find_server_metrics_json(job_dir: Path) -> Path | None:
    """Find the server_metrics_export.json file for a srtslurm job."""
    json_files = list(job_dir.glob("logs/artifacts/*/server_metrics_export.json"))
    json_files = [f for f in json_files if "warmup" not in str(f)]
    
    if json_files:
        return json_files[0]
    return None


def get_metric_stat_aggregated(metrics: dict, metric_name: str, stat: str, agg: str = "first") -> float | None:
    """Extract a stat from a metric, aggregating across all series (workers).
    
    agg options:
        "first" - return first series value (default)
        "sum" - sum across all series
        "max" - max across all series  
        "avg" - average across all series
    """
    metric = metrics.get(metric_name)
    if not metric:
        return None
    series = metric.get("series", [])
    if not series:
        return None
    
    values = []
    for s in series:
        val = s.get("stats", {}).get(stat)
        if val is not None:
            values.append(val)
    
    if not values:
        return None
    
    if agg == "first":
        return values[0]
    elif agg == "sum":
        return sum(values)
    elif agg == "max":
        return max(values)
    elif agg == "avg":
        return sum(values) / len(values)
    return values[0]


def extract_kv_cache_metrics(json_path: Path) -> Dict[str, object]:
    """Extract KV cache metrics from server_metrics_export.json.
    
    Aggregates across all workers:
    - total_blocks: average across workers (capacity is similar per worker)
    - util_max: max across all workers (worst case utilization)
    - reused/missed blocks: sum across workers (total cache activity)
    - hit_rate: calculated from summed reused/(reused+missed)
    """
    result = {
        "kv_total_blocks": None,
        "kv_blocksize": None,
        "kv_util_max": None,
        "kv_reused_blocks": None,
        "kv_missed_blocks": None,
        "kv_hit_rate": None,
    }
    
    if not json_path or not json_path.exists():
        return result
    
    try:
        with json_path.open() as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        
        # Total blocks - average across workers (they have similar capacity)
        result["kv_total_blocks"] = get_metric_stat_aggregated(
            metrics, "dynamo_component_total_blocks", "avg", agg="avg"
        )
        
        # Block size in tokens (constant per model)
        result["kv_blocksize"] = get_metric_stat_aggregated(
            metrics, "dynamo_frontend_model_kv_cache_block_size", "avg", agg="first"
        )
        
        # Max utilization - take max across all workers
        result["kv_util_max"] = get_metric_stat_aggregated(
            metrics, "trtllm_kv_cache_utilization", "max", agg="max"
        )
        
        # Cumulative counters - sum across all workers
        result["kv_reused_blocks"] = get_metric_stat_aggregated(
            metrics, "trtllm_kv_cache_reused_blocks", "total", agg="sum"
        )
        result["kv_missed_blocks"] = get_metric_stat_aggregated(
            metrics, "trtllm_kv_cache_missed_blocks", "total", agg="sum"
        )
        
        # Calculate hit rate from summed reused / (reused + missed)
        reused = result["kv_reused_blocks"]
        missed = result["kv_missed_blocks"]
        if reused is not None and missed is not None:
            total = reused + missed
            if total > 0:
                result["kv_hit_rate"] = reused / total
    except Exception:
        pass
    
    return result


def calculate_kv_workspace_gib(
    total_blocks: float | None, blocksize: float | None, cache_mb_per_1k: float
) -> float | None:
    """Calculate total KV cache workspace in GiB (binary).
    
    Formula: total_tokens = total_blocks * blocksize
             total_1k_seqs = total_tokens / 1000
             total_cache_GiB = (total_1k_seqs * cache_mb_per_1k) / 1024
    """
    if total_blocks is None or blocksize is None:
        return None
    total_tokens = total_blocks * blocksize
    total_1k_seqs = total_tokens / 1000
    total_cache_mb = total_1k_seqs * cache_mb_per_1k
    return total_cache_mb / 1024


def check_runtime_errors(job_dir: Path) -> str:
    """Check worker logs for runtime errors.
    
    Returns:
        "v" if no errors found, otherwise "[ErrorType (details)]"
    """
    import re
    import subprocess
    
    logs_dir = job_dir / "logs"
    if not logs_dir.exists():
        return "v"
    
    # Search for common errors in .out files
    error_patterns = [
        (r"AssertionError: total_num_tokens \((\d+)\) should be less than or equal to max_num_tokens \((\d+)\)",
         lambda m: f"[AssertionError (total_num_tokens {m.group(1)} > max_num_tokens {m.group(2)})]"),
        (r"AssertionError: (.{1,50})",
         lambda m: f"[AssertionError ({m.group(1).strip()})]"),
        (r"torch\.AcceleratorError: CUDA error: (.{1,50})",
         lambda m: f"[CUDA error ({m.group(1).strip()})]"),
        (r"RuntimeError: (.{1,50})",
         lambda m: f"[RuntimeError ({m.group(1).strip()})]"),
        (r"OutOfMemoryError|out of memory|OOM",
         lambda m: "[OOM]"),
    ]
    
    try:
        # Read all .out files
        for out_file in logs_dir.glob("*.out"):
            if out_file.name == "benchmark.out":
                continue
            try:
                content = out_file.read_text(errors="ignore")
                for pattern, formatter in error_patterns:
                    match = re.search(pattern, content)
                    if match:
                        return formatter(match)
            except Exception:
                continue
    except Exception:
        pass
    
    return "v"


def row_from_srtslurm_job(job_dir: Path, cache_mb_per_1k: float = 34.31) -> Dict[str, object] | None:
    """Extract a row of stats from a srtslurm job directory."""
    json_path = find_srtslurm_aiperf_json(job_dir)
    if not json_path:
        print(f"Warning: No profile_export_aiperf.json found in {job_dir}", file=sys.stderr)
        return None
    
    info = extract_srtslurm_info(job_dir)
    
    # Extract KV cache metrics from server_metrics_export.json
    server_metrics_path = find_server_metrics_json(job_dir)
    kv_metrics = extract_kv_cache_metrics(server_metrics_path)
    
    with json_path.open() as f:
        data = json.load(f)
    
    request_count = parse_float(data.get("request_count", {}).get("avg"))
    error_count = parse_float(data.get("error_request_count", {}).get("avg"))
    if error_count is None:
        error_count = 0.0
    
    ttft = data.get("time_to_first_token", {})
    itl = data.get("inter_token_latency", {})
    
    total_token_tput = parse_float(data.get("total_token_throughput", {}).get("avg"))
    request_tput = parse_float(data.get("request_throughput", {}).get("avg"))
    goodput = parse_float(data.get("goodput", {}).get("avg"))
    output_tput_per_user = parse_float(data.get("output_token_throughput_per_user", {}).get("avg"))
    
    # Calculate derived metrics
    total_token_tput_per_gpu = None
    if total_token_tput and info["gpus"]:
        total_token_tput_per_gpu = total_token_tput / info["gpus"]
    
    error_rate_pct = 0.0
    if request_count:
        total = request_count + error_count
        if total > 0:
            error_rate_pct = (error_count / total) * 100
    
    # Combined error field: "count [pct%]"
    error_count_int = int(error_count) if error_count else 0
    errors_combined = f"{error_count_int} [{error_rate_pct:.1f}%]"
    
    # Check for runtime errors in worker logs
    runtime_error = check_runtime_errors(job_dir)
    
    return {
        "dataset": info["dataset"],
        "srtslurm_id": info["srtslurm_id"],
        "config_name": info["config_name"],
        "concurrency": info["concurrency"],
        "request_count": request_count,
        "errors [pct]": errors_combined,
        "runtime_error": runtime_error,
        "ttft_avg_ms": parse_float(ttft.get("avg")),
        "ttft_p50_ms": parse_float(ttft.get("p50")),
        "ttft_p99_ms": parse_float(ttft.get("p99")),
        "itl_avg_ms": parse_float(itl.get("avg")),
        "itl_p50_ms": parse_float(itl.get("p50")),
        "itl_p99_ms": parse_float(itl.get("p99")),
        "output_tput_per_user": output_tput_per_user,
        "total_token_tput": total_token_tput,
        "request_tput": request_tput,
        "goodput": goodput,
        "total_token_tput_per_gpu": total_token_tput_per_gpu,
        "kv_total_blocks (dynamo_component_total_blocks)": kv_metrics["kv_total_blocks"],
        "kv_blocksize (dynamo_frontend_model_kv_cache_block_size)": kv_metrics["kv_blocksize"],
        "kv_total_workspace_GiB (calculated)": calculate_kv_workspace_gib(
            kv_metrics["kv_total_blocks"], kv_metrics["kv_blocksize"], cache_mb_per_1k
        ),
        "kv_util_max (trtllm_kv_cache_utilization)": kv_metrics["kv_util_max"],
        "kv_reused_blocks (trtllm_kv_cache_reused_blocks)": kv_metrics["kv_reused_blocks"],
        "kv_missed_blocks (trtllm_kv_cache_missed_blocks)": kv_metrics["kv_missed_blocks"],
        "kv_hit_rate (calculated)": kv_metrics["kv_hit_rate"],
    }


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def find_profile_files(run_root: Path) -> Iterable[Path]:
    """Find profile_export_aiperf files (CSV or JSON) in run_root.
    
    Prefers JSON over CSV when both exist in the same directory.
    """
    patterns = [
        "artifacts/*/trace_replay_c*/profile_export_aiperf",
        "artifacts/*/concurrency_*/profile_export_aiperf",
    ]
    files = []
    for pattern in patterns:
        for base in run_root.glob(pattern + ".json"):
            files.append(base)
        for base in run_root.glob(pattern + ".csv"):
            json_path = base.with_suffix(".json")
            if not json_path.exists():
                files.append(base)
    return sorted(set(files))


def concurrency_from_path(file_path: Path) -> int:
    """Extract concurrency value from directory name (trace_replay_c* or concurrency_*)."""
    name = file_path.parent.name
    if name.startswith("trace_replay_c"):
        return int(name.removeprefix("trace_replay_c"))
    elif name.startswith("concurrency_"):
        return int(name.removeprefix("concurrency_"))
    raise ValueError(f"Cannot extract concurrency from path: {file_path}")


def profile_jsonl_path(csv_path: Path) -> Path:
    return csv_path.with_name("profile_export.jsonl")


def load_sla_stats(jsonl_path: Path) -> Dict[str, int]:
    good_request_count = 0

    with jsonl_path.open() as f:
        for line in f:
            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                continue

            ttft_metric = metrics.get("time_to_first_token")
            itl_metric = metrics.get("inter_token_latency")
            if not isinstance(ttft_metric, dict) or not isinstance(itl_metric, dict):
                continue

            ttft_value = ttft_metric.get("value")
            itl_value = itl_metric.get("value")
            if ttft_value is None or itl_value is None:
                continue

            if float(ttft_value) < SLA_TTFT_MS and float(itl_value) < SLA_ITL_MS:
                good_request_count += 1

    return {"good_request_count": good_request_count}


def row_from_csv(run_root: Path, csv_path: Path) -> Dict[str, object]:
    """Parse metrics from a CSV file."""
    metrics = load_metrics(csv_path)
    sla_stats = load_sla_stats(profile_jsonl_path(csv_path))

    request_count = parse_int(metrics[METRIC_REQUEST_COUNT]["avg"])
    error_metric = metrics.get(METRIC_ERROR_REQUEST_COUNT)
    error_count = parse_int(error_metric["avg"]) if error_metric else 0
    request_throughput = parse_float(metrics[METRIC_REQUEST_TPUT]["avg"])
    good_request_count = sla_stats["good_request_count"]

    ttft = metrics[METRIC_TTFT]
    sla_goodput_pct = None
    good_request_throughput = None
    if request_count:
        sla_goodput_pct = (good_request_count / request_count) * 100.0
        if request_throughput is not None:
            good_request_throughput = request_throughput * (good_request_count / request_count)

    return {
        "dataset": run_root.name,
        "concurrency": concurrency_from_path(csv_path),
        "request_count": request_count,
        "error_count": error_count,
        "ttft_avg_ms": parse_float(ttft["avg"]),
        "ttft_p50_ms": parse_float(ttft["p50"]),
        "ttft_p90_ms": parse_float(ttft["p90"]),
        "ttft_p99_ms": parse_float(ttft["p99"]),
        "output_tput_per_user": parse_float(metrics[METRIC_OUTPUT_TPUT_PER_USER]["avg"]),
        "total_token_tput": parse_float(metrics[METRIC_TOTAL_TOKEN_TPUT]["avg"]),
        "request_tput_rps": request_throughput,
        "good_request_count": good_request_count,
        "sla_goodput_pct": sla_goodput_pct,
        "good_request_tput_rps": good_request_throughput,
    }


def row_from_json(dataset_name: str, json_path: Path) -> Dict[str, object]:
    """Parse metrics from a JSON file."""
    with json_path.open() as f:
        data = json.load(f)

    request_count = parse_int(data.get("request_count", {}).get("avg"))
    good_request_count = parse_int(data.get("good_request_count", {}).get("avg"))
    request_throughput = parse_float(data.get("request_throughput", {}).get("avg"))
    goodput = parse_float(data.get("goodput", {}).get("avg"))

    ttft = data.get("time_to_first_token", {})

    sla_goodput_pct = None
    if request_count and good_request_count is not None:
        sla_goodput_pct = (good_request_count / request_count) * 100.0

    return {
        "dataset": dataset_name,
        "concurrency": concurrency_from_path(json_path),
        "request_count": request_count,
        "error_count": 0,
        "ttft_avg_ms": parse_float(ttft.get("avg")),
        "ttft_p50_ms": parse_float(ttft.get("p50")),
        "ttft_p90_ms": parse_float(ttft.get("p90")),
        "ttft_p99_ms": parse_float(ttft.get("p99")),
        "output_tput_per_user": parse_float(data.get("output_token_throughput_per_user", {}).get("avg")),
        "total_token_tput": parse_float(data.get("total_token_throughput", {}).get("avg")),
        "request_tput_rps": request_throughput,
        "good_request_count": good_request_count,
        "sla_goodput_pct": sla_goodput_pct,
        "good_request_tput_rps": goodput,
    }


def row_from_file(dataset_name: str, file_path: Path) -> Dict[str, object]:
    """Parse metrics from a CSV or JSON file."""
    if file_path.suffix == ".json":
        return row_from_json(dataset_name, file_path)
    else:
        # For CSV, we need the run_root which is the parent of 'artifacts'
        parts = file_path.parts
        if "artifacts" in parts:
            artifacts_idx = parts.index("artifacts")
            run_root = Path(*parts[:artifacts_idx])
        else:
            run_root = file_path.parent.parent.parent
        return row_from_csv(run_root, file_path)


def collect_rows(paths: Iterable[Path]) -> List[Dict[str, object]]:
    """Collect rows from run roots or direct file paths."""
    rows: List[Dict[str, object]] = []
    for path in paths:
        if path.is_file():
            # Direct file path - use parent directory name as dataset
            dataset_name = path.parent.parent.name
            rows.append(row_from_file(dataset_name, path))
        else:
            # Run root directory - find all profile files
            for file_path in find_profile_files(path):
                rows.append(row_from_file(path.name, file_path))
    return sorted(rows, key=lambda row: (str(row["dataset"]), int(row["concurrency"])))


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "concurrency",
        "request_count",
        "error_count",
        "ttft_avg_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "output_tput_per_user",
        "total_token_tput",
        "request_tput_rps",
        "good_request_count",
        "sla_goodput_pct",
        "good_request_tput_rps",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_csv(output_path: Path) -> None:
    print(output_path.read_text(), end="")


def format_tsv_value(val) -> str:
    """Format a value for TSV output."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val == 0:
            return "0"
        elif abs(val) < 0.01:
            return f"{val:.6f}"
        elif abs(val) < 1:
            return f"{val:.4f}"
        elif abs(val) < 100:
            return f"{val:.2f}"
        else:
            return f"{val:.1f}"
    return str(val)


def print_tsv_header(delimiter: str = "\t") -> None:
    """Print header row."""
    print(delimiter.join(TSV_COLUMNS))


def print_tsv_row(row: Dict[str, object], delimiter: str = "\t") -> None:
    """Print a single row."""
    values = [format_tsv_value(row.get(col)) for col in TSV_COLUMNS]
    print(delimiter.join(values))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", 
        nargs="*", 
        help="Run folders or profile_export_aiperf files (CSV/JSON) to parse"
    )
    parser.add_argument(
        "--dir", "-d",
        nargs="+",
        dest="job_ids",
        help="srtslurm job IDs to look up in outputs directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="parsed-aiperf-trace-replay.csv",
        help="Output CSV path (ignored with --tsv)",
    )
    parser.add_argument(
        "--tsv", "-t",
        action="store_true",
        help="Output tab-separated values to stdout (for Excel paste)"
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Print header row only"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output comma-separated values instead of tab-separated"
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(SRTSLURM_OUTPUTS_DIR),
        help=f"srtslurm outputs directory (default: {SRTSLURM_OUTPUTS_DIR})"
    )
    parser.add_argument(
        "--cache-mb-per-1k",
        type=float,
        default=34.31,
        help="KV cache size in MB per 1K sequence length (default: 34.31 for Kimi-K2)"
    )
    args = parser.parse_args()

    # Determine delimiter
    delimiter = "," if args.csv else "\t"

    # Handle --header (print header, then continue if --dir is also specified)
    if args.header:
        print_tsv_header(delimiter)
        if not args.job_ids and not args.paths:
            return 0

    # Collect rows from job IDs
    srtslurm_rows: List[Dict[str, object]] = []
    if args.job_ids:
        outputs_dir = Path(args.outputs_dir)
        for job_id in args.job_ids:
            job_dir = find_srtslurm_job_dir(job_id, outputs_dir)
            if job_dir:
                row = row_from_srtslurm_job(job_dir, cache_mb_per_1k=args.cache_mb_per_1k)
                if row:
                    srtslurm_rows.append(row)
            else:
                print(f"Warning: Job directory not found for {job_id}", file=sys.stderr)

    # Collect rows from direct paths
    path_rows: List[Dict[str, object]] = []
    if args.paths:
        paths = [Path(p) for p in args.paths]
        path_rows = collect_rows(paths)

    # Combine and output
    all_rows = srtslurm_rows + path_rows

    if not all_rows:
        print("No data found.", file=sys.stderr)
        return 1

    if args.tsv or args.job_ids or args.csv:
        # TSV/CSV output mode (default for --dir)
        for row in all_rows:
            print_tsv_row(row, delimiter)
    else:
        # CSV file output mode
        output_path = Path(args.output)
        write_csv(path_rows, output_path)
        print_csv(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())