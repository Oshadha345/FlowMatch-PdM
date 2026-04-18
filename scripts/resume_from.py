#!/usr/bin/env python3
"""FlowMatch-PdM — Reset a pipeline step back to 'pending' for re-execution.

Usage:
  python scripts/resume_from.py --phase 0 --track bearing_rul --dataset FEMTO
  python scripts/resume_from.py --phase 1 --aug noise --track bearing_rul --dataset XJTU-SY
  python scripts/resume_from.py --phase 3 --model COTGAN --track bearing_rul --dataset FEMTO --step gen
  python scripts/resume_from.py --phase 3 --model FlowMatch --track bearing_rul --dataset FEMTO --step gen
  python scripts/resume_from.py --phase 4 --ablation no_prior --track bearing_rul --dataset FEMTO --step gen
  python scripts/resume_from.py --phase 4 --ablation no_lap --track bearing_rul --dataset XJTU-SY --step both
  python scripts/resume_from.py --phase final_report
  python scripts/resume_from.py --phase env_check
  python scripts/resume_from.py --phase preflight

Add --force to reset entries that are already 'done'.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "pipeline_state.json"


def load_state() -> dict:
    if not STATE_FILE.exists():
        print(f"ERROR: {STATE_FILE} does not exist.")
        sys.exit(1)
    with STATE_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    tmp.replace(STATE_FILE)


def confirm(message: str) -> bool:
    print(message)
    resp = input("Proceed? [y/N] ").strip().lower()
    return resp == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Reset a pipeline step to 'pending' for re-execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", required=True,
                        help="Phase to reset: 0, 1, 2, 3, 4, env_check, preflight, final_report")
    parser.add_argument("--track", default=None, help="Track: bearing_rul")
    parser.add_argument("--dataset", default=None, help="Dataset name: FEMTO or XJTU-SY")
    parser.add_argument("--model", default=None, help="Generator model name (Phase 2/3)")
    parser.add_argument("--aug", default=None, help="Augmentation type (Phase 1): noise or smote")
    parser.add_argument("--ablation", default=None, help="Ablation variant (Phase 4): no_prior, no_tccm, or no_lap")
    parser.add_argument("--step", default="both", choices=["gen", "clf", "both"],
                        help="Which step to reset for Phase 2/3/4 (default: both)")
    parser.add_argument("--force", action="store_true",
                        help="Force reset even if currently 'done'")
    args = parser.parse_args()

    state = load_state()

    phase = args.phase

    # --- Simple scalar resets ---
    if phase == "env_check":
        current = state.get("env_check", "pending")
        if current == "done" and not args.force:
            print(f"env_check is already 'done'. Use --force to reset.")
            sys.exit(0)
        if confirm(f"Reset env_check from '{current}' to 'pending'?"):
            state["env_check"] = "pending"
            save_state(state)
            print("Done.")
        return

    if phase == "preflight":
        current = state.get("preflight_notebook", "pending")
        if current == "done" and not args.force:
            print(f"preflight_notebook is already 'done'. Use --force to reset.")
            sys.exit(0)
        if confirm(f"Reset preflight_notebook from '{current}' to 'pending'?"):
            state["preflight_notebook"] = "pending"
            save_state(state)
            print("Done.")
        return

    if phase == "final_report":
        current = state.get("final_report", "pending")
        if current == "done" and not args.force:
            print(f"final_report is already 'done'. Use --force to reset.")
            sys.exit(0)
        if confirm(f"Reset final_report from '{current}' to 'pending'?"):
            state["final_report"] = "pending"
            save_state(state)
            print("Done.")
        return

    # --- Phase 0 ---
    if phase == "0":
        if not args.track or not args.dataset:
            print("ERROR: --track and --dataset required for Phase 0")
            sys.exit(1)
        key = f"{args.track}__{args.dataset}"
        entry = state.get("phase0", {}).get(key)
        if entry is None:
            print(f"ERROR: Key '{key}' not found in phase0")
            sys.exit(1)
        current = entry.get("status", "pending")
        if current == "done" and not args.force:
            print(f"phase0.{key} is already 'done'. Use --force to reset.")
            sys.exit(0)
        if confirm(f"Reset phase0.{key} from '{current}' to 'pending'?"):
            entry["status"] = "pending"
            entry["run_id"] = None
            for k in list(entry.keys()):
                if k not in ("status", "run_id"):
                    del entry[k]
            save_state(state)
            print("Done.")
        return

    # --- Phase 1 ---
    if phase == "1":
        if not args.aug or not args.track or not args.dataset:
            print("ERROR: --aug, --track, and --dataset required for Phase 1")
            sys.exit(1)
        key = f"{args.aug}__{args.track}__{args.dataset}"
        entry = state.get("phase1", {}).get(key)
        if entry is None:
            print(f"ERROR: Key '{key}' not found in phase1")
            sys.exit(1)
        current = entry.get("status", "pending")
        if current == "done" and not args.force:
            print(f"phase1.{key} is already 'done'. Use --force to reset.")
            sys.exit(0)
        if confirm(f"Reset phase1.{key} from '{current}' to 'pending'?"):
            entry["status"] = "pending"
            entry["run_id"] = None
            for k in list(entry.keys()):
                if k not in ("status", "run_id"):
                    del entry[k]
            save_state(state)
            print("Done.")
        return

    # --- Phase 2 / 3 ---
    if phase in ("2", "3"):
        if not args.model or not args.track or not args.dataset:
            print(f"ERROR: --model, --track, and --dataset required for Phase {phase}")
            sys.exit(1)
        key = f"{args.model}__{args.track}__{args.dataset}"
        phase_key = f"phase{phase}"
        entry = state.get(phase_key, {}).get(key)
        if entry is None:
            print(f"ERROR: Key '{key}' not found in {phase_key}")
            sys.exit(1)

        changes = []
        if args.step in ("gen", "both"):
            current_gen = entry.get("gen_status", "pending")
            if current_gen == "done" and not args.force:
                print(f"{phase_key}.{key}.gen_status is 'done'. Use --force to reset.")
            else:
                changes.append(("gen_status", current_gen))
        if args.step in ("clf", "both"):
            current_clf = entry.get("clf_status", "pending")
            if current_clf == "done" and not args.force:
                print(f"{phase_key}.{key}.clf_status is 'done'. Use --force to reset.")
            else:
                changes.append(("clf_status", current_clf))

        if not changes:
            print("Nothing to reset.")
            sys.exit(0)

        desc = ", ".join(f"{k}: '{v}' -> 'pending'" for k, v in changes)
        if confirm(f"Reset {phase_key}.{key}: {desc}?"):
            for field, _ in changes:
                entry[field] = "pending"
                id_field = field.replace("_status", "_run_id")
                entry[id_field] = None
                # Clear associated metrics
                if field == "gen_status":
                    for mk in ["ftsd", "mmd", "discriminative_score", "predictive_score_mae"]:
                        entry.pop(mk, None)
                if field == "clf_status":
                    for mk in ["clf_rmse", "clf_f1_macro"]:
                        entry.pop(mk, None)
            save_state(state)
            print("Done.")
        return

    # --- Phase 4 ---
    if phase == "4":
        if not args.ablation or not args.track or not args.dataset:
            print("ERROR: --ablation, --track, and --dataset required for Phase 4")
            sys.exit(1)
        key = f"FlowMatch_{args.ablation}__{args.track}__{args.dataset}"
        entry = state.get("phase4", {}).get(key)
        if entry is None:
            print(f"ERROR: Key '{key}' not found in phase4")
            sys.exit(1)

        changes = []
        if args.step in ("gen", "both"):
            current_gen = entry.get("gen_status", "pending")
            if current_gen == "done" and not args.force:
                print(f"phase4.{key}.gen_status is 'done'. Use --force to reset.")
            else:
                changes.append(("gen_status", current_gen))
        if args.step in ("clf", "both"):
            current_clf = entry.get("clf_status", "pending")
            if current_clf == "done" and not args.force:
                print(f"phase4.{key}.clf_status is 'done'. Use --force to reset.")
            else:
                changes.append(("clf_status", current_clf))

        if not changes:
            print("Nothing to reset.")
            sys.exit(0)

        desc = ", ".join(f"{k}: '{v}' -> 'pending'" for k, v in changes)
        if confirm(f"Reset phase4.{key}: {desc}?"):
            for field, _ in changes:
                entry[field] = "pending"
                id_field = field.replace("_status", "_run_id")
                entry[id_field] = None
                if field == "gen_status":
                    for mk in ["ftsd", "mmd", "discriminative_score", "predictive_score_mae"]:
                        entry.pop(mk, None)
                if field == "clf_status":
                    for mk in ["clf_rmse", "clf_f1_macro"]:
                        entry.pop(mk, None)
            save_state(state)
            print("Done.")
        return

    print(f"ERROR: Unknown phase '{phase}'. Valid: 0, 1, 2, 3, 4, env_check, preflight, final_report")
    sys.exit(1)


if __name__ == "__main__":
    main()
