#!/usr/bin/env python3
"""Sequential non-FlowMatch queue for the active FEMTO/XJTU-SY RUL pivot."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
NON_FLOWMATCH_EXECUTION = REPO_ROOT / "non_flowmatch_execution.sh"


def main() -> int:
    if not NON_FLOWMATCH_EXECUTION.exists():
        print(f"Missing runner: {NON_FLOWMATCH_EXECUTION}", file=sys.stderr)
        return 1

    print("orchestrate.py now runs only the non-FlowMatch queue for the active bearing_rul pivot.")
    result = subprocess.run(["bash", str(NON_FLOWMATCH_EXECUTION)], cwd=REPO_ROOT)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
