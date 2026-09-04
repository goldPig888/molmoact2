#!/usr/bin/env python3
"""Live terminal dashboard for a MolmoAct2 VLAReplica training run."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path


STEP_RE = re.compile(r"\[step=(\d+)/(\d+), eta=([^\]]+)\]")
METRIC_RE = re.compile(r"^\s*([\w./-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*$", re.I)
EVAL_RE = re.compile(r"Running evaluation for '([^']+)'|Eval for '([^']+)' done")


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_rows() -> list[list[str]]:
    output = run_text([
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    return [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]


def parse_log(path: Path) -> dict[str, object]:
    state: dict[str, object] = {"step": 0, "total": 0, "eta": "waiting", "metrics": {}, "events": []}
    if not path.exists():
        return state
    lines = path.read_text(errors="replace").splitlines()
    metrics: dict[str, str] = {}
    current_section = ""
    events: list[str] = []
    for line in lines:
        if match := STEP_RE.search(line):
            state.update(step=int(match.group(1)), total=int(match.group(2)), eta=match.group(3))
            current_section = "train"
        if match := EVAL_RE.search(line):
            current_section = match.group(1) or match.group(2) or "validation"
        if metric := METRIC_RE.match(line):
            key, value = metric.groups()
            metrics[f"{current_section}/{key}" if current_section else key] = value
        lowered = line.lower()
        if any(word in lowered for word in ("warning", "error", "traceback", "outofmemory")):
            events.append(line.strip())
    state["metrics"] = metrics
    state["events"] = events[-5:]
    state["complete"] = any("Training complete" in line for line in lines)
    return state


def format_bytes(value: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="/metadisk/may/molmoact2-checkpoints/pilot-200-live.log")
    parser.add_argument("--save-folder", default="/metadisk/may/molmoact2-checkpoints/pilot-200-live")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Render once and exit (useful for checks).")
    args = parser.parse_args()
    log_path = Path(args.log)
    save_folder = Path(args.save_folder)
    started = time.monotonic()

    try:
        while True:
            state = parse_log(log_path)
            step, total = int(state["step"]), int(state["total"])
            percent = (100 * step / total) if total else 0.0
            width = 36
            filled = round(width * percent / 100)
            bar = "#" * filled + "-" * (width - filled)
            processes = run_text(["pgrep", "-af", "train_lerobot.py"]).splitlines()
            checkpoints = sorted(p.name for p in save_folder.glob("step*") if p.is_dir()) if save_folder.exists() else []
            folder_size = 0
            if save_folder.exists():
                disk = run_text(["du", "-sb", str(save_folder)]).split()
                folder_size = int(disk[0]) if disk and disk[0].isdigit() else 0
            usage = shutil.disk_usage("/metadisk") if Path("/metadisk").exists() else None

            print("\033[2J\033[H", end="")
            print("MolmoAct2 · VLAReplica live fine-tuning")
            print("=" * 72)
            status = "COMPLETE" if state.get("complete") else ("RUNNING" if processes else "WAITING / STOPPED")
            print(f"Status: {status:<18} Elapsed: {timedelta(seconds=int(time.monotonic()-started))}")
            print(f"Progress: [{bar}] {step}/{total or '?'} ({percent:5.1f}%)  ETA: {state['eta']}")
            print("\nGPUs")
            print(" ID  Name                 Util    VRAM                 Temp   Power")
            for row in gpu_rows():
                if len(row) >= 7:
                    idx, name, util, used, maximum, temp, power = row[:7]
                    print(f" {idx:>2}  {name[:20]:<20} {util:>3}%   {used:>6}/{maximum:<6} MiB  {temp:>3} C  {float(power):5.1f} W")

            print("\nLatest metrics")
            metrics = state["metrics"]
            wanted = [k for k in metrics if any(x in k.lower() for x in ("loss", "grad_norm", "throughput", "tokens_per"))]
            if wanted:
                for key in wanted[-12:]:
                    print(f" {key:<54} {metrics[key]}")
            else:
                print(" Waiting for the first logged training step...")

            print("\nOutputs")
            print(f" Folder: {save_folder}")
            print(f" Size: {format_bytes(folder_size)}   Checkpoints: {', '.join(checkpoints) or 'none'}")
            if usage:
                print(f" Metadisk free: {format_bytes(usage.free)} / {format_bytes(usage.total)}")
            print(f" Worker processes: {len(processes)}")

            print("\nRecent warnings/errors")
            events = state["events"]
            if events:
                for event in events:
                    print(" " + event[-160:])
            else:
                print(" none")
            print("\nCtrl+C closes only this dashboard; it does not stop training.", flush=True)
            if args.once:
                break
            time.sleep(max(args.refresh, 0.5))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
