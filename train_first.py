#!/usr/bin/env python3
"""Cross-platform train-first launcher. Never scans VarWISE targets."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    print("Checking VarWise's offline safety tests...", flush=True)
    check = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests",
         "-p", "test*.py", "-q"], cwd=ROOT, check=False
    )
    if check.returncode:
        print("STOP: safety tests failed; training not started.", flush=True)
        return check.returncode
    extras = sys.argv[1:]
    if "--out" not in extras:
        extras = ["--out", "training_only"] + extras
    print("Starting real Gaia/ZTF training ONLY. Validation remains sealed until fit.", flush=True)
    command = [sys.executable, "-u", str(ROOT / "src" / "varwise_gaia_hunt.py"),
               "train", *extras]
    proc = subprocess.run(command, cwd=ROOT, check=False)
    if proc.returncode:
        print("NOT VALIDATED. Discovery remains LOCKED. No VarWISE targets were queried.", flush=True)
    else:
        print("Train-first validation passed. Review the validation report before searching.", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
