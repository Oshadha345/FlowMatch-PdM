#!/usr/bin/env python3
"""FlowMatch-PdM — Pipeline progress checker.

Reads pipeline_state.json and prints a live progress table.
Can be run in a second terminal without interfering with the orchestrator.

Usage:
    python scripts/check_results.py
    python scripts/check_results.py --watch   # refresh every 60 seconds
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "pipeline_state.json"

GENERATORS = [
    "FlowMatch", "FaultDiffusion", "COTGAN", "DiffusionTS", "TimeFlow",
]

RUL_DATASETS = [
    ("bearing_rul", "FEMTO"),
    ("bearing_rul", "XJTU-SY"),
]


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    with STATE_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _status_icon(status: str) -> str:
    if status == "done":
        return "\033[92m✓\033[0m"
    if status == "failed":
        return "\033[91m✗\033[0m"
    if status == "pending":
        return "\033[93m·\033[0m"
    return "?"


def _fmt(val, prec: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{prec}f}"
    return str(val)


def _pad(text: str, width: int) -> str:
    # Account for ANSI escape codes in width calculation
    visible_len = len(text.encode("utf-8").decode("unicode_escape")) if "\033" not in text else len(text) - text.count("\033[") * 4 - text.count("m") + text.count("\033[")
    # Simple approach
    stripped = text
    for code in ["\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"]:
        stripped = stripped.replace(code, "")
    pad_needed = max(0, width - len(stripped))
    return text + " " * pad_needed


def print_status(state: dict) -> None:
    """Print the full status dashboard."""
    print("\033[1mFlowMatch-PdM — Pipeline Status\033[0m")
    print("=" * 72)
    print(f"  Last updated: {state.get('last_updated', 'unknown')}")
    print()

    # Environment
    env = state.get("env_check", "pending")
    print(f"  ENV CHECK:       {_status_icon(env)} {env}")

    # Datasets
    acq = state.get("dataset_acquisition", {})
    ds_parts = []
    for ds in ["FEMTO", "XJTU-SY"]:
        s = acq.get(ds, "pending")
        ds_parts.append(f"{ds} {_status_icon(s)}")
    print(f"  DATASETS:        {' | '.join(ds_parts)}")

    # Preflight
    pf = state.get("preflight_notebook", "pending")
    print(f"  PREFLIGHT:       {_status_icon(pf)} {pf}")
    print()

    # Phase 0
    print("\033[1m  PHASE 0 — BASELINES\033[0m")
    print("  ┌─────────────────────────────────┬──────────┬──────────────┐")
    print("  │ Dataset                         │ Status   │ Score        │")
    print("  ├─────────────────────────────────┼──────────┼──────────────┤")
    phase0 = state.get("phase0", {})
    all_p0 = [
        ("FEMTO", "bearing_rul", "RMSE"),
        ("XJTU-SY", "bearing_rul", "RMSE"),
    ]
    for ds, track, metric in all_p0:
        key = f"{track}__{ds}"
        entry = phase0.get(key, {})
        status = entry.get("status", "pending")
        metric_key = "rmse" if "rul" in track else "f1_macro"
        score = _fmt(entry.get(metric_key))
        ds_label = f"{ds} ({metric})"
        print(f"  │ {ds_label:<31} │ {_pad(_status_icon(status) + ' ' + status, 8)} │ {score:<12} │")
    print("  └─────────────────────────────────┴──────────┴──────────────┘")
    print()

    # Phase 1
    print("\033[1m  PHASE 1 — CLASSICAL AUGMENTATION\033[0m")
    phase1 = state.get("phase1", {})
    p1_done = sum(1 for v in phase1.values() if isinstance(v, dict) and v.get("status") == "done")
    p1_total = len(phase1)
    print(f"  ({p1_done} / {p1_total} complete)")
    print("  ┌──────────────────────────────────┬──────────┬──────────────┐")
    print("  │ Aug / Dataset                    │ Status   │ Score        │")
    print("  ├──────────────────────────────────┼──────────┼──────────────┤")
    for k, v in phase1.items():
        if not isinstance(v, dict):
            continue
        status = v.get("status", "pending")
        metric_val = v.get("rmse", v.get("f1_macro"))
        score = _fmt(metric_val)
        print(f"  │ {k:<32} │ {_pad(_status_icon(status) + ' ' + status, 8)} │ {score:<12} │")
    print("  └──────────────────────────────────┴──────────┴──────────────┘")
    print()

    # Phase 3
    phase3 = state.get("phase3", {})
    if phase3:
        p3_done = sum(1 for v in phase3.values() if isinstance(v, dict) and v.get("clf_status") == "done")
        print(f"\033[1m  PHASE 3 — GENERATORS\033[0m  ({p3_done}/{len(phase3)} complete)")
        print("  ┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("  │ Model            │ Dataset  │ Gen      │ FTSD     │ MMD      │ CLF      │")
        print("  ├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
        for gen in GENERATORS:
            for track, ds in RUL_DATASETS:
                key = f"{gen}__{track}__{ds}"
                entry = phase3.get(key, {})
                gen_status = entry.get("gen_status", "pending")
                clf_status = entry.get("clf_status", "pending")
                ftsd = _fmt(entry.get("ftsd"))
                mmd = _fmt(entry.get("mmd"))
                print(f"  │ {gen:<16} │ {ds:<8} │ {_pad(_status_icon(gen_status) + ' ' + gen_status[:4], 8)} │ "
                      f"{ftsd:<8} │ {mmd:<8} │ {_pad(_status_icon(clf_status) + ' ' + clf_status[:4], 8)} │")
        print("  └──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
        print()

    # Top models
    top3 = state.get("top_models", [])
    if top3:
        labels = " | ".join(f"#{i+1} {m}" for i, m in enumerate(top3))
        print(f"  \033[1mTOP MODELS:\033[0m  {labels}")
        print()

    # Phase 4
    phase4 = state.get("phase4", {})
    if phase4:
        p4_done = sum(1 for v in phase4.values() if isinstance(v, dict) and v.get("gen_status") == "done")
        print(f"\033[1m  PHASE 4 — ABLATIONS\033[0m  ({p4_done}/{len(phase4)} complete)")
        for k in sorted(phase4):
            v = phase4[k]
            if not isinstance(v, dict):
                continue
            gs = v.get("gen_status", "pending")
            cs = v.get("clf_status", "pending")
            ftsd = _fmt(v.get("ftsd"))
            print(f"    {k}: gen={_status_icon(gs)} clf={_status_icon(cs)} ftsd={ftsd}")
        print()

    # Final report
    fr = state.get("final_report", "pending")
    print(f"  FINAL REPORT:    {_status_icon(fr)} {fr}")
    print()

    # Errors
    errors = state.get("errors", [])
    if errors:
        print(f"  \033[91mERRORS ({len(errors)}):\033[0m")
        for err in errors[-5:]:
            print(f"    [{err.get('timestamp', '?')}] {err.get('key_path', '?')}: {err.get('error', '?')[:100]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Check FlowMatch-PdM pipeline status")
    parser.add_argument("--watch", action="store_true", help="Refresh every 60 seconds")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                state = _load_state()
                if state:
                    print_status(state)
                else:
                    print("No pipeline_state.json found.")
                print(f"  [Refreshing every 60s — Ctrl-C to stop]")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        state = _load_state()
        if state:
            print_status(state)
        else:
            print("No pipeline_state.json found. Run 'bash launch.sh' to start the pipeline.")
            sys.exit(0)


if __name__ == "__main__":
    main()
